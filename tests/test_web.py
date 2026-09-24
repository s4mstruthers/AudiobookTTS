"""The web API, driven through FastAPI's test client."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient
from helpers import needs_ffmpeg, tiny_png, write_tone

from audiobooktts import config, web
from audiobooktts.store import JobStore, Manifest


@pytest.fixture
def client():
    # The default app only accepts localhost host names; the test client uses
    # "testserver", so build one without that restriction.
    with TestClient(web.create_app(restrict_hosts=False)) as c:
        yield c


def _wait_until_finished(client, job_id: str, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(j for j in client.get("/api/jobs").json()["jobs"] if j["job_id"] == job_id)
        if not job["live"]:
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def _events(client, job_id: str) -> list[dict]:
    with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
        return [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]


def test_index_and_health(client):
    assert "AudioBookTTS" in client.get("/").text
    assert client.get("/api/health").json()["status"] == "ok"


def test_upload_parses_the_book(client, epub_path):
    r = client.post("/api/upload", files={"file": ("book.epub", epub_path.read_bytes())})
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "The Test Book" and body["has_cover"]
    assert [c["title"] for c in body["chapters"]] == ["Chapter I", "Chapter II", "Chapter III"]
    # Stored by content: a different book with the same file name cannot
    # overwrite it.
    assert "0123456789abcdef" not in body["path"]
    assert client.get("/api/cover", params={"path": body["path"]}).content == tiny_png()


def test_upload_rejects_non_epub(client, tmp_path):
    r = client.post("/api/upload", files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 400
    r = client.post("/api/upload", files={"file": ("broken.epub", b"not a zip")})
    assert r.status_code == 400 and "epub" in r.json()["detail"].lower()
    assert not any(config.UPLOAD_DIR.rglob("broken.epub"))


def test_book_paths_must_be_uploads(client, epub_path):
    # Arbitrary files on disk are not served or parsed.
    for url in ("/api/cover", "/api/pronunciation/suggest"):
        assert client.get(url, params={"path": str(epub_path)}).status_code == 400
    r = client.post("/api/jobs", json={"path": str(epub_path), "voice": "default"})
    assert r.status_code == 400


def test_cover_upload_is_content_addressed(client):
    r = client.post("/api/cover", files={"file": ("c.png", tiny_png())})
    assert r.status_code == 200
    cover = r.json()
    assert client.get(cover["url"]).content == tiny_png()
    # The old endpoint served any file on disk given its path.
    assert client.get("/api/cover-file", params={"path": "/etc/passwd"}).status_code == 404
    assert client.get("/api/covers/..%2F..%2Fconfig.json").status_code == 404
    assert client.post("/api/cover", files={"file": ("c.gif", b"GIF89a")}).status_code == 400


@needs_ffmpeg
def test_preview_returns_audio_and_sample(client, uploaded_epub):
    r = client.get(
        "/api/preview", params={"path": str(uploaded_epub), "voice": "default", "chapter": 1}
    )
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert "Holmes" in r.headers["x-sample-text"]


def test_preview_unknown_voice_is_a_clear_400(client, uploaded_epub):
    r = client.get("/api/preview", params={"path": str(uploaded_epub), "voice": "nobody"})
    assert r.status_code == 400 and "Unknown voice" in r.json()["detail"]


@needs_ffmpeg
def test_job_lifecycle(client, uploaded_epub, tmp_path):
    r = client.post(
        "/api/jobs",
        json={"path": str(uploaded_epub), "voice": "default", "output_dir": str(tmp_path / "out")},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    events = _events(client, job_id)
    assert events[-1]["stage"] == "done"
    assert any(e["stage"] == "synthesizing" for e in events)

    job = _wait_until_finished(client, job_id)
    # Regression: finished jobs stayed "live" forever, showing Watch/Cancel
    # instead of Download, and could never be deleted.
    assert job["status"] == "done" and not job["live"]

    # A client that attaches after the end still learns the outcome.
    assert _events(client, job_id) == [{"stage": "done", "message": job["output_path"]}]

    download = client.get(f"/api/jobs/{job_id}/download")
    assert download.status_code == 200 and download.content[4:8] == b"ftyp"

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/download").status_code == 404


def test_job_rejects_unknown_voice_and_engine(client, uploaded_epub):
    bad_voice = client.post("/api/jobs", json={"path": str(uploaded_epub), "voice": "nobody"})
    assert bad_voice.status_code == 400
    bad_engine = client.post(
        "/api/jobs", json={"path": str(uploaded_epub), "voice": "default", "engine": "espeak"}
    )
    assert bad_engine.status_code == 400
    bad_speed = client.post(
        "/api/jobs", json={"path": str(uploaded_epub), "voice": "default", "speed": 9}
    )
    assert bad_speed.status_code == 422


def test_unknown_and_unsafe_job_ids(client):
    for job_id in ("nope-000000", "..", "UPPER"):
        assert client.post(f"/api/jobs/{job_id}/resume").status_code == 404
        assert client.delete(f"/api/jobs/{job_id}").status_code == 404
        assert client.get(f"/api/jobs/{job_id}/events").status_code == 404
    assert client.post("/api/jobs/nope-000000/cancel").status_code == 404


def test_resume_refuses_a_job_running_elsewhere(client):
    import os

    store = JobStore()
    store.save(Manifest(
        job_id="busy-000001", epub_path="", epub_sha256="", book_title="B", book_author="A",
        engine="chatterbox", voice="", speed=1.0, status="running", pid=os.getppid(),
    ))
    assert client.post("/api/jobs/busy-000001/resume").status_code == 409
    assert client.delete("/api/jobs/busy-000001").status_code == 409


def test_pronunciation_endpoints(client):
    assert client.post("/api/pronunciation", json={"word": "O'Brien", "say_as": "Oh Bry-en"}).ok
    entries = client.get("/api/pronunciation").json()["entries"]
    assert entries == [{"word": "O'Brien", "say_as": "Oh Bry-en"}]
    assert client.post("/api/pronunciation", json={"word": "", "say_as": "x"}).status_code == 422
    r = client.get("/api/pronunciation/try", params={"word": "O'Brien", "voice": "default"})
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert client.delete("/api/pronunciation/O'Brien").ok
    assert client.get("/api/pronunciation").json()["entries"] == []


@needs_ffmpeg
def test_voice_library_flow(client, tmp_path):
    recording = write_tone(tmp_path / "reading.wav", 40)
    r = client.post(
        "/api/voice-source",
        files={"file": ("reading.wav", recording.read_bytes())},
        data={"name": "Irene Adler!"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "Irene_Adler"
    assert r.json()["duration"] == pytest.approx(40, abs=0.1)

    assert client.post("/api/voice-library/Irene_Adler/recut", json={"start": 5, "duration": 12}).ok
    library = client.get("/api/voice-library").json()
    assert [v["name"] for v in library["voices"]] == ["Irene_Adler"]
    assert library["voices"][0]["seconds"] == pytest.approx(12, abs=0.2)
    assert library["windows"]["speaker_s"] == 15

    assert client.get("/api/voice-library/Irene_Adler/audio").content[:4] == b"RIFF"
    segment = client.get("/api/voice-source/Irene_Adler/segment", params={"start": 1, "duration": 3})
    assert segment.content[:4] == b"RIFF"
    voices = client.get("/api/voices").json()["voices"]
    assert {"id": "Irene_Adler", "kind": "clone"}.items() <= voices[1].items()

    assert client.delete("/api/voice-library/Irene_Adler").ok
    assert client.get("/api/voice-library").json()["voices"] == []
    assert client.delete("/api/voice-library/Irene_Adler").status_code == 404


def test_voice_names_are_validated(client, tmp_path):
    for bad in ("", "!!!", "default"):
        r = client.post(
            "/api/voice-source", files={"file": ("a.wav", b"RIFF")}, data={"name": bad}
        )
        assert r.status_code == 400


def test_browse_lists_folders(client, tmp_path):
    (tmp_path / "Audiobooks").mkdir()
    (tmp_path / ".hidden").mkdir()
    listing = client.get("/api/browse", params={"path": str(tmp_path)}).json()
    assert [d["name"] for d in listing["dirs"]] == ["Audiobooks"]
    assert listing["parent"] == str(tmp_path.resolve().parent)


def test_cross_origin_writes_are_refused(client):
    r = client.post("/api/jobs/x-000000/cancel", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    same = client.post("/api/jobs/x-000000/cancel", headers={"Origin": "http://testserver"})
    assert same.status_code == 404  # allowed through, then not found


def test_default_app_only_answers_to_localhost():
    with TestClient(web.create_app(), base_url="http://127.0.0.1:8765") as local:
        assert local.get("/api/health").status_code == 200
    with TestClient(web.create_app(), base_url="http://attacker.example") as rebound:
        # DNS rebinding: a hostile name resolving to 127.0.0.1 is rejected.
        assert rebound.get("/api/health").status_code == 400


def test_missing_backend_is_reported_as_503(client, uploaded_epub, monkeypatch):
    from audiobooktts import engines
    from audiobooktts.engines import EngineUnavailableError

    class NoBackend:
        name = "chatterbox"
        sample_rate = 24000
        deterministic = False

        def list_voices(self):
            return []

        def conditioning_windows(self):
            return None

        def load(self):
            raise EngineUnavailableError("No Chatterbox backend is installed.")

        synthesize = load

    monkeypatch.setattr(engines, "_build", lambda name: NoBackend())
    engines.reset_engines()
    r = client.get("/api/preview", params={"path": str(uploaded_epub), "voice": "default"})
    assert r.status_code == 503 and "backend" in r.json()["detail"]
