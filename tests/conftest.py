"""Shared fixtures.

Every test runs against a private data directory and a synthetic TTS engine,
so the suite needs no GPU, no model download and never touches real user data.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

# Must be set before audiobooktts is imported: data paths are fixed at import.
_HOME = Path(tempfile.mkdtemp(prefix="abtts-test-home-"))
os.environ["AUDIOBOOKTTS_HOME"] = str(_HOME)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from audiobooktts import config  # noqa: E402
from audiobooktts import engines as engines_mod  # noqa: E402
from audiobooktts.engines.base import CloningEngine  # noqa: E402
from helpers import build_epub  # noqa: E402


class FakeEngine(CloningEngine):
    """Deterministic stand-in for Chatterbox: a tone whose length tracks the words."""

    name = "chatterbox"
    sample_rate = 24000
    deterministic = False
    seconds_per_word = 0.35

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def conditioning_windows(self):
        return {"speaker_s": 15.0, "decoder_s": 10.0, "useful_s": 15.0}

    def synthesize(self, text, voice, speed=1.0):
        from audiobooktts.voices import reference_path

        reference_path(voice)  # unknown voices fail, as with the real engine
        self.calls.append((text, voice))
        n = int(self.sample_rate * self.seconds_per_word * max(1, len(text.split())))
        t = np.arange(n) / self.sample_rate
        tone = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
        pad = np.zeros(int(self.sample_rate * 0.3))
        return np.concatenate([pad, tone, pad]).astype(np.float32)


@pytest.fixture(autouse=True)
def clean_home(monkeypatch):
    """A fresh, empty data directory and a fake engine for every test."""
    for child in _HOME.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    fake = FakeEngine()
    monkeypatch.setattr(engines_mod, "_build", lambda name: fake)
    engines_mod.reset_engines()
    yield
    engines_mod.reset_engines()


@pytest.fixture
def fake_engine():
    return engines_mod.get_engine("chatterbox").inner


@pytest.fixture
def app_home() -> Path:
    return config.APP_DIR


@pytest.fixture
def epub_path(tmp_path) -> Path:
    return build_epub(tmp_path / "book.epub")


@pytest.fixture
def uploaded_epub(tmp_path) -> Path:
    """A test epub placed where the web UI keeps uploads."""
    dest = config.UPLOAD_DIR / "0123456789abcdef" / "book.epub"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return build_epub(dest)
