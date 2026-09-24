"""User configuration and application data locations.

All state lives under one directory, ``~/.audiobooktts`` by default. Set the
``AUDIOBOOKTTS_HOME`` environment variable to keep it somewhere else.
"""

from __future__ import annotations

import logging
import os
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path

from audiobooktts.fsutil import read_json, write_json

logger = logging.getLogger(__name__)


def _app_dir() -> Path:
    override = os.environ.get("AUDIOBOOKTTS_HOME")
    return Path(override).expanduser() if override else Path.home() / ".audiobooktts"


APP_DIR = _app_dir()
CONFIG_PATH = APP_DIR / "config.json"
JOBS_DIR = APP_DIR / "jobs"
#: Reference clips: every audio file here is a selectable cloned voice.
VOICES_DIR = APP_DIR / "voices"
#: The full recordings voices were cut from, kept so a clip can be re-cut.
VOICE_SOURCES_DIR = APP_DIR / "voice_sources"
UPLOAD_DIR = APP_DIR / "uploads"
COVER_DIR = APP_DIR / "covers"
LEXICON_PATH = APP_DIR / "pronunciation.json"
PACING_CACHE_PATH = APP_DIR / "voice_pacing.json"

DEFAULT_CHATTERBOX_MODEL = "mlx-community/chatterbox-turbo-8bit"


@dataclass
class Config:
    engine: str = "chatterbox"
    # Each engine has its own voice set, so the choice is remembered per engine
    # rather than being clobbered when you switch. Empty means "pick a sensible
    # one for this engine".
    voice: str = ""
    voice_by_engine: dict[str, str] = field(default_factory=dict)
    speed: float = 1.0
    bitrate: str = "80k"
    output_dir: str = str(Path.home() / "Audiobooks")

    # Pause lengths in seconds. The engine's own leading and trailing padding is
    # trimmed from every chunk first, so these alone set the pacing.
    gap_clause: float = 0.25  # mid-sentence, where a long sentence was split
    # Only applied where a long paragraph had to be split across calls; set to
    # roughly the engine's own sentence gap so the join is inaudible.
    gap_sentence: float = 0.3
    gap_paragraph: float = 1.9  # a real beat between paragraphs
    gap_title: float = 2.6  # after the announced chapter title

    # Off by default: the engine's own rhythm between sentences sounds right,
    # and forcing every full stop to a fixed length made the reading laboured.
    # Paragraph breaks are still inserted, which is where a beat is wanted.
    stretch_sentence_pauses: bool = False
    # "paragraph" reads a whole paragraph in one call so it flows, then
    # lengthens the gaps the engine left between its sentences. "sentence"
    # generates each sentence separately: exact pauses, but each one starts
    # cold and the delivery sounds disconnected.
    synthesis_unit: str = "paragraph"
    # Match every voice to one narration pace. Voices otherwise differ by over
    # 10%, and most run faster than a commercial audiobook. 0 disables it.
    target_wpm: float = 160.0
    # Master the finished book to ACX loudness (-19 LUFS, peaks under -3 dB).
    master_audio: bool = True

    # Which Chatterbox weights to load. With the MLX backend this is a Hugging
    # Face repo id; "turbo" variants are distilled and fast, the base model is
    # slower but reproduces a voice more closely. The PyTorch backend follows
    # the same choice: a name containing "turbo" selects Chatterbox-Turbo.
    chatterbox_model: str = DEFAULT_CHATTERBOX_MODEL
    # "auto", "mlx" (Apple Silicon) or "torch" (Windows, Linux, Intel Macs).
    chatterbox_backend: str = "auto"
    # PyTorch device: "auto", "cuda", "mps" or "cpu".
    device: str = "auto"

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Read the saved config, falling back to defaults for anything unusable.

        A hand-edited file with a typo or a wrongly typed value must not stop
        the program from starting, so bad entries are reported and skipped.
        """
        path = path or CONFIG_PATH
        if not path.exists():
            return cls()
        data = read_json(path)
        if not isinstance(data, dict):
            logger.warning("Ignoring unreadable config file %s; using defaults", path)
            return cls()

        values = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            default = f.default if f.default is not MISSING else f.default_factory()
            value = _coerce(data[f.name], default)
            if value is None:
                logger.warning("Ignoring invalid config value %s=%r", f.name, data[f.name])
            else:
                values[f.name] = value
        # Unknown keys (e.g. "llm" from older versions) are ignored.
        return cls(**values)

    def save(self, path: Path | None = None) -> None:
        write_json(path or CONFIG_PATH, asdict(self))

    def voice_for(self, engine: str | None = None) -> str:
        """The remembered voice for an engine, falling back to its own default.

        A voice from one engine may mean nothing to another, so asking for one
        engine's voice while another is selected has to resolve to something
        that engine actually has.
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


def _coerce(value, default):
    """Convert a JSON value to the type of ``default``, or None if impossible."""
    if isinstance(default, bool):
        return value if isinstance(value, bool) else None
    if isinstance(default, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    if isinstance(default, str):
        return value if isinstance(value, str) else None
    if isinstance(default, dict):
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        return None
    return value
