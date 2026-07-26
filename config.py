"""User configuration and application data directories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_DIR = Path.home() / ".audiobooktts"
JOBS_DIR = APP_DIR / "jobs"
CONFIG_PATH = APP_DIR / "config.json"


@dataclass
class LLMConfig:
    provider: str = "none"  # none | ollama | anthropic
    model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"


@dataclass
class Config:
    engine: str = "kokoro"
    voice: str = "af_heart"
    speed: float = 1.0
    bitrate: str = "80k"
    output_dir: str = str(Path.home() / "Audiobooks")
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            llm = LLMConfig(**data.pop("llm", {}))
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(llm=llm, **known)
        return cls()

    def save(self) -> None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))
