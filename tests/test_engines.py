"""Backend selection and the PyTorch Chatterbox adapter, with the ML stack faked."""

from __future__ import annotations

import os
import sys
import time
import types

import numpy as np
import pytest
from helpers import write_tone

from audiobooktts import config
from audiobooktts.engines import EngineUnavailableError
from audiobooktts.engines import chatterbox as cb


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeModel:
    """Mimics chatterbox.tts.ChatterboxTTS: conds attribute, prepare, generate."""

    instances: list[_FakeModel] = []

    def __init__(self, device):
        self.device = device
        self.sr = 24000
        self.conds = "builtin-conds"
        self.prepared: list[str] = []
        self.generated_with: list[object] = []
        _FakeModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, device):
        return cls(device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        self.prepared.append(wav_fpath)
        self.conds = f"conds:{os.path.basename(wav_fpath)}:{len(self.prepared)}"

    def generate(self, text):
        self.generated_with.append(self.conds)
        return _FakeTensor(np.ones((1, 2400)))


@pytest.fixture
def fake_torch_stack(monkeypatch):
    """Install fake `torch` and `chatterbox` packages for the duration of a test."""
    _FakeModel.instances.clear()
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch.inference_mode = _NoGrad
    chatterbox = types.ModuleType("chatterbox")
    tts = types.ModuleType("chatterbox.tts")
    tts.ChatterboxTTS = _FakeModel
    turbo = types.ModuleType("chatterbox.tts_turbo")
    turbo.ChatterboxTurboTTS = _FakeModel
    for name, module in {
        "torch": torch, "chatterbox": chatterbox,
        "chatterbox.tts": tts, "chatterbox.tts_turbo": turbo,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(cb, "installed_backends", lambda: {"mlx": False, "torch": True})
    return torch


def test_auto_prefers_mlx_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(cb, "installed_backends", lambda: {"mlx": True, "torch": True})
    monkeypatch.setattr(cb, "is_apple_silicon", lambda: True)
    assert cb.resolve_backend("auto") == "mlx"
    monkeypatch.setattr(cb, "is_apple_silicon", lambda: False)
    assert cb.resolve_backend("auto") == "torch"


def test_missing_backend_explains_how_to_install(monkeypatch):
    monkeypatch.setattr(cb, "installed_backends", lambda: {"mlx": False, "torch": False})
    with pytest.raises(EngineUnavailableError, match="pip install"):
        cb.resolve_backend("auto")
    with pytest.raises(EngineUnavailableError, match="chatterbox-tts"):
        cb.resolve_backend("torch")
    with pytest.raises(EngineUnavailableError, match="Unknown"):
        cb.resolve_backend("tensorflow")


def test_engine_constructs_without_any_backend(monkeypatch):
    # Listing voices and managing the library must work before installing ML.
    monkeypatch.setattr(cb, "installed_backends", lambda: {"mlx": False, "torch": False})
    engine = cb.ChatterboxEngine()
    assert [v.id for v in engine.list_voices()] == ["default"]
    with pytest.raises(EngineUnavailableError):
        engine.synthesize("Hello there.", "default")


def test_device_resolution(fake_torch_stack):
    assert cb.resolve_device("auto") == "cpu"
    fake_torch_stack.cuda.is_available = lambda: True
    assert cb.resolve_device("auto") == "cuda"
    fake_torch_stack.cuda.is_available = lambda: False
    with pytest.raises(EngineUnavailableError, match="CUDA"):
        cb.resolve_device("cuda")


def test_torch_backend_switches_and_caches_voices(fake_torch_stack):
    write_tone(config.VOICES_DIR / "alice.wav", 8)
    write_tone(config.VOICES_DIR / "bob.wav", 8)
    engine = cb.ChatterboxEngine(backend="torch", device="auto")

    out = engine.synthesize("Hello there.", "alice")
    assert out.dtype == np.float32 and out.shape == (2400,)
    engine.synthesize("Again.", "alice")
    engine.synthesize("Now bob.", "bob")
    engine.synthesize("Built-in voice.", "default")
    engine.synthesize("Alice again.", "alice")

    model = _FakeModel.instances[-1]
    # Each clip is prepared once, however often the voice changes.
    assert [os.path.basename(p) for p in model.prepared] == ["alice.wav", "bob.wav"]
    assert model.generated_with == [
        "conds:alice.wav:1", "conds:alice.wav:1", "conds:bob.wav:2",
        "builtin-conds", "conds:alice.wav:1",
    ]
    assert engine.describe() == "Chatterbox-Turbo on torch (cpu)"


def test_recut_clip_is_prepared_again(fake_torch_stack):
    # Regression: conditioning was cached by voice name, so a re-cut clip kept
    # its old sound until the program restarted.
    clip = write_tone(config.VOICES_DIR / "alice.wav", 8)
    engine = cb.ChatterboxEngine(backend="torch")
    engine.synthesize("One.", "alice")
    time.sleep(0.01)
    write_tone(clip, 9)
    engine.synthesize("Two.", "alice")
    assert len(_FakeModel.instances[-1].prepared) == 2


def test_non_native_clip_is_converted_for_torch(fake_torch_stack):
    pytest.importorskip("soundfile")
    from helpers import HAS_FFMPEG

    if not HAS_FFMPEG:
        pytest.skip("ffmpeg is not installed")
    import subprocess

    wav = write_tone(config.APP_DIR / "src.wav", 8)
    m4a = config.VOICES_DIR / "carol.m4a"
    m4a.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(m4a)], check=True)
    engine = cb.ChatterboxEngine(backend="torch")
    engine.synthesize("Hello.", "carol")
    prepared = _FakeModel.instances[-1].prepared[0]
    assert prepared.endswith(".wav") and not os.path.exists(prepared)  # temp file cleaned up


def test_unknown_voice_is_an_error(fake_torch_stack):
    from audiobooktts.voices import VoiceClipError

    engine = cb.ChatterboxEngine(backend="torch")
    with pytest.raises(VoiceClipError, match="Unknown voice"):
        engine.synthesize("Hello.", "nobody")


def test_conditioning_windows_follow_the_variant():
    assert cb.ChatterboxEngine(model_id="x/chatterbox-turbo").conditioning_windows()["speaker_s"] == 15
    assert cb.ChatterboxEngine(model_id="x/chatterbox-fp16").conditioning_windows()["speaker_s"] == 6
