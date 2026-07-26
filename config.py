"""User configuration and application data directories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_DIR = Path.home() / ".audiobooktts"
JOBS_DIR = APP_DIR / "jobs"
CONFIG_PATH = APP_DIR / "config.json"


@dataclass
class Config:
    engine: str = "chatterbox"
    # Each engine has its own voice set, so the choice is remembered per engine
    # rather than being clobbered when you switch. Empty means "pick a sensible
    # one for this engine".
    voice: str = ""
    voice_by_engine: dict[str, str] = field(
        default_factory=lambda: {"kokoro": "bf_emma"}
    )
    speed: float = 1.0
    bitrate: str = "80k"
    output_dir: str = str(Path.home() / "Audiobooks")
    # Pause lengths in seconds. Every sentence is synthesised separately so
    # these fully determine the pacing; the engine's own sentence gap (~0.13s)
    # is trimmed away first. Tune to taste.
    gap_clause: float = 0.2    # mid-sentence, where a long sentence was split
    gap_sentence: float = 0.65  # after a full stop
    gap_paragraph: float = 1.1
    gap_title: float = 1.8
    # Match every voice to one narration pace. Voices otherwise differ by over
    # 10%, and most run faster than a commercial audiobook. 0 disables it.
    target_wpm: float = 160.0
    # Master the finished book to ACX loudness (-19 LUFS, peaks under -3 dB).
    master_audio: bool = True

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            # Unknown keys (e.g. "llm" from older versions) are ignored.
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        return cls()

    def save(self) -> None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))

    def voice_for(self, engine: str | None = None) -> str:
        """The remembered voice for an engine, falling back to its own default.

        A Kokoro voice id means nothing to Chatterbox and vice versa, so asking
        for one engine's voice while another is selected has to resolve to
        something that engine actually has.
        """
        engine = engine or self.engine
        remembered = self.voice_by_engine.get(engine) or (
            self.voice if engine == self.engine else ""
        )
        from audiobooktts.engines import get_engine

        available = [v.id for v in get_engine(engine).list_voices()]
        if remembered in available:
            return remembered
        # Prefer a cloned voice over a model's generic built-in one.
        clones = [v for v in available if v != "default"]
        return (clones or available or [""])[0]

    def remember_voice(self, engine: str, voice: str) -> None:
        self.engine = engine
        self.voice = voice
        self.voice_by_engine[engine] = voice
