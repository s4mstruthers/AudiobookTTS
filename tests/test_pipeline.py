"""Job storage, filesystem helpers and the conversion pipeline end to end."""

from __future__ import annotations

import os
import subprocess
import threading

import pytest
import soundfile as sf

from audiobooktts import config, pacing
from audiobooktts.config import Config
from audiobooktts.fsutil import atomic_write_text, is_within, safe_filename
from audiobooktts.mastering import meets_acx, verify
from audiobooktts.pipeline import JobCancelledError, create_job, run_job
from audiobooktts.store import JobNotFoundError, JobStore, Manifest, pid_alive
from audiobooktts.voices import VoiceClipError
from helpers import build_epub, needs_ffmpeg, tiny_png, write_tone

# --- filesystem and config ---------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ('Why? A "Tale": 1/2', "Why A Tale 12"),
        ("Ends with dots...", "Ends with dots"),
        ("CON", "_CON"),
        ("", "untitled"),
        ("Zoë & Chloé", "Zoë & Chloé"),
    ],
)
def test_safe_filename(name, expected):
    assert safe_filename(name) == expected


def test_atomic_write_is_utf8_and_leaves_no_temp_files(tmp_path):
    target = tmp_path / "sub" / "f.json"
    atomic_write_text(target, "Сергей — naïve")
    atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"
    assert [p.name for p in target.parent.iterdir()] == ["f.json"]


def test_is_within(tmp_path):
    assert is_within(tmp_path / "a" / "b", tmp_path)
    assert not is_within(tmp_path / ".." / "elsewhere", tmp_path)


def test_config_survives_bad_values(app_home):
    (app_home / "config.json").write_text(
        '{"speed": "fast", "voice": "alice", "master_audio": false, "old_key": 1}', encoding="utf-8"
    )
    cfg = Config.load()
    assert (cfg.speed, cfg.voice, cfg.master_audio) == (1.0, "alice", False)
    (app_home / "config.json").write_text("{not json", encoding="utf-8")
    assert Config.load() == Config()


# --- store -------------------------------------------------------------------


def _manifest(job_id: str = "book-abc123", **kw) -> Manifest:
    base = dict(
        job_id=job_id,
        epub_path="",
        epub_sha256="",
        book_title="B",
        book_author="A",
        engine="chatterbox",
        voice="",
        speed=1.0,
    )
    return Manifest(**{**base, **kw})


@pytest.mark.parametrize("bad_id", ["..", "../x", "a/b", "", "UPPER", "x" * 200])
def test_store_rejects_unsafe_job_ids(bad_id):
    # Regression: delete("..") removed the whole data directory.
    store = JobStore()
    with pytest.raises(JobNotFoundError):
        store.delete(bad_id)
    with pytest.raises(JobNotFoundError):
        store.load(bad_id)


def test_reconcile_only_touches_dead_processes():
    store = JobStore()
    store.save(_manifest("dead-000001", status="running", pid=2**22 + 12345))
    store.save(_manifest("mine-000001", status="running", pid=os.getpid()))
    store.save(_manifest("done-000001", status="done"))
    assert store.reconcile_running() == ["dead-000001"]
    assert store.load("dead-000001").status == "interrupted"
    assert store.load("mine-000001").status == "running"


def test_pid_alive():
    assert pid_alive(os.getpid())
    assert not pid_alive(0)


def test_manifest_ignores_unknown_fields():
    data = _manifest().to_dict()
    data["future_field"] = 1
    data["chapters"] = [{"index": 0, "title": "T", "extra": True}]
    assert Manifest.from_dict(data).chapters[0].title == "T"


# --- pipeline ------------------------------------------------------------------


def _config(tmp_path, **kw) -> Config:
    cfg = Config(output_dir=str(tmp_path / "out"), voice="default", **kw)
    return cfg


@needs_ffmpeg
def test_convert_end_to_end(tmp_path, epub_path, fake_engine):
    cfg = _config(tmp_path)
    manifest = create_job(epub_path, cfg)
    stages = []
    done = run_job(manifest.job_id, cfg, on_progress=lambda p: stages.append(p.stage))

    assert done.status == "done" and done.pid == 0
    assert stages[0] == "parsing" and stages[-1] == "done"
    assert {"loading", "calibrating", "synthesizing", "packaging"} <= set(stages)
    assert fake_engine.loaded

    out = tmp_path / "out" / "The Test Book - Jane Tester.m4b"
    assert done.output_path == str(out) and out.is_file()
    assert not list(out.parent.glob("*.part"))

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-show_streams", "-of", "json", str(out)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert probe.count('"title": "Chapter') == 3
    assert '"attached_pic": 1' in probe  # the epub's cover was embedded

    # Mastered to ACX loudness.
    assert meets_acx(verify(out))
    # Every chapter duration matches its FLAC.
    for ch in done.chapters:
        flac = JobStore().chapter_audio_path(done.job_id, ch.index)
        assert ch.duration_s == pytest.approx(sf.info(str(flac)).duration)


@needs_ffmpeg
def test_pace_is_measured_once_and_reused(tmp_path, epub_path, fake_engine):
    cfg = _config(tmp_path)
    m = run_job(create_job(epub_path, cfg).job_id, cfg)
    assert pacing.is_calibrated(fake_engine, "default")
    tempo = m.tempo_by_voice["default"]
    # The fake reads at 0.35 s per word (~171 wpm), so it is slowed towards 160.
    assert 0.85 < tempo < 1.0
    calls = len(fake_engine.calls)
    run_job(create_job(epub_path, cfg).job_id, cfg)
    calibration_calls = sum(1 for text, _ in fake_engine.calls[calls:] if "evening train" in text)
    assert calibration_calls == 0


@needs_ffmpeg
def test_resume_skips_finished_chapters(tmp_path, epub_path, fake_engine):
    cfg = _config(tmp_path, target_wpm=0)
    manifest = create_job(epub_path, cfg)
    cancel = threading.Event()

    def stop_in_second_chapter(p):
        if p.stage == "synthesizing" and p.chapter_index == 1:
            cancel.set()

    with pytest.raises(JobCancelledError):
        run_job(manifest.job_id, cfg, on_progress=stop_in_second_chapter, cancel_event=cancel)
    store = JobStore()
    assert store.load(manifest.job_id).status == "cancelled"
    assert store.load(manifest.job_id).chapters_done == 1

    fake_engine.calls.clear()
    done = run_job(manifest.job_id, cfg)
    assert done.status == "done" and done.chapters_done == 3
    assert not any(text.startswith("Chapter 1.") for text, _ in fake_engine.calls)


@needs_ffmpeg
def test_custom_cover_is_copied_into_the_job(tmp_path, epub_path):
    cover = tmp_path / "mine.png"
    cover.write_bytes(tiny_png(16, 16))
    cfg = _config(tmp_path, target_wpm=0)
    manifest = create_job(epub_path, cfg, cover_path=cover)
    cover.unlink()  # the job must not depend on the original file
    assert run_job(manifest.job_id, cfg).status == "done"


def test_missing_voice_fails_before_rendering(tmp_path, epub_path, fake_engine):
    cfg = _config(tmp_path)
    cfg.voice = "nobody"
    manifest = create_job(epub_path, cfg)
    with pytest.raises(VoiceClipError):
        run_job(manifest.job_id, cfg)
    assert JobStore().load(manifest.job_id).status == "failed"
    assert fake_engine.calls == []


def test_changed_epub_is_refused(tmp_path):
    path = build_epub(tmp_path / "b.epub")
    cfg = _config(tmp_path)
    manifest = create_job(path, cfg)
    build_epub(path, title="Different")
    with pytest.raises(RuntimeError, match="changed"):
        run_job(manifest.job_id, cfg)


@needs_ffmpeg
def test_per_chapter_voices(tmp_path, epub_path, fake_engine):
    write_tone(config.VOICES_DIR / "narrator2.wav", 8)
    cfg = _config(tmp_path, target_wpm=0)
    manifest = create_job(epub_path, cfg, chapter_voices={2: "narrator2"})
    run_job(manifest.job_id, cfg)
    voices_used = {voice for text, voice in fake_engine.calls if text.startswith("Chapter 3")}
    assert voices_used == {"narrator2"}
