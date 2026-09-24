"""Master the finished narration to audiobook loudness standards.

Raw TTS output measures around -26 dB RMS with peaks near -2 dB: too quiet on
average yet too hot at the peaks, which is why it sounds thin next to a
commercial audiobook. Levels also differ between engines, so chapters would not
match each other.

The targets come from ACX, Audible's submission standard: RMS between -23 and
-18 dB, peaks no higher than -3 dB, noise floor below -60 dB. Loudness is
measured with EBU R128 in a first pass and corrected in a second, which is far
more accurate than single-pass normalisation and does not pump.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from audiobooktts import ffmpeg

logger = logging.getLogger(__name__)

# Mid-ACX, and the usual target for spoken-word audiobooks.
TARGET_LUFS = -19.0
ACX_PEAK_DB = -3.0
ACX_RMS_RANGE_DB = (-23.0, -18.0)
# Aim below the ceiling: limiting exactly at -3.0 lands around -2.9993, which
# fails the spec by a whisker.
TARGET_PEAK_DB = -3.3
TARGET_LRA = 7.0

# Rumble and DC offset carry no speech but eat headroom.
HIGHPASS_HZ = 65
# Gentle: enough to tame TTS sibilance without lisping the output.
DEESSER_INTENSITY = 0.12

# atempo accepts 0.5-2.0 per stage.
_ATEMPO_MIN, _ATEMPO_MAX = 0.5, 2.0


class MasteringError(RuntimeError):
    pass


def atempo_chain(tempo: float) -> str:
    """A pitch-preserving time-stretch, chained for shifts beyond one stage."""
    if tempo <= 0:
        raise ValueError(f"tempo must be positive, got {tempo}")
    stages = []
    remaining = float(tempo)
    while remaining < _ATEMPO_MIN:
        stages.append(f"atempo={_ATEMPO_MIN}")
        remaining /= _ATEMPO_MIN
    while remaining > _ATEMPO_MAX:
        stages.append(f"atempo={_ATEMPO_MAX}")
        remaining /= _ATEMPO_MAX
    stages.append(f"atempo={remaining:.6f}")
    return ",".join(stages)


def apply_filters(audio: np.ndarray, sample_rate: int, filters: str) -> np.ndarray:
    """Run an ffmpeg filter chain over an in-memory mono float32 array.

    Filtering is an enhancement, so a failure logs a warning and returns the
    input unchanged rather than losing the narration.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not filters:
        return audio
    raw = ["-f", "f32le", "-ar", str(sample_rate), "-ac", "1"]
    proc = ffmpeg.run(
        [
            "-hide_banner",
            "-loglevel",
            "error",
            *raw,
            "-i",
            "pipe:0",
            "-af",
            filters,
            *raw,
            "pipe:1",
        ],
        input=audio.tobytes(),
    )
    if proc.returncode != 0 or not proc.stdout:
        logger.warning(
            "ffmpeg filter %r failed; using unfiltered audio: %s",
            filters,
            proc.stderr.decode("utf-8", "replace")[-500:],
        )
        return audio
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def time_stretch_file(src: Path, dst: Path, tempo: float) -> None:
    """Time-stretch an audio file into a FLAC at ``dst``. Raises on failure."""
    proc = ffmpeg.run(
        [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-af",
            atempo_chain(tempo),
            "-c:a",
            "flac",
            "-sample_fmt",
            "s16",
            "-f",
            "flac",
            str(dst),
        ],
        text=True,
    )
    if proc.returncode != 0:
        raise MasteringError(f"Time-stretch failed: {proc.stderr[-500:]}")


def pre_filters() -> list[str]:
    """Clean-up stages that run before loudness correction."""
    return [
        f"highpass=f={HIGHPASS_HZ}",
        f"deesser=i={DEESSER_INTENSITY}",
        # Light 2:1 — TTS is already consistent (measured 2-4 LU range), so
        # this tightens rather than squashes.
        "acompressor=threshold=-18dB:ratio=2:attack=15:release=250:makeup=1",
    ]


def measure(path: str | Path, input_args: list[str] | None = None) -> dict[str, float] | None:
    """First pass: measure loudness. None if ffmpeg gives nothing usable.

    The clean-up stages run here too. Loudness correction is computed from this
    measurement but applied after those stages, so measuring the raw signal
    instead left the result well short of target — the compressor alone moved
    it by more than 2 LU.

    input_args lets the caller pass demuxer flags, e.g. the concat list the
    packager assembles rather than a plain audio file.
    """
    chain = ",".join(
        [
            *pre_filters(),
            f"loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK_DB}:LRA={TARGET_LRA}:print_format=json",
        ]
    )
    result = ffmpeg.run(
        [
            "-hide_banner",
            "-nostats",
            *(input_args or []),
            "-i",
            str(path),
            "-af",
            chain,
            "-f",
            "null",
            "-",
        ],
        text=True,
    )
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.S)
    if not match:
        logger.warning("Loudness measurement failed; falling back to single-pass mastering")
        return None
    try:
        data = json.loads(match.group(0))
        return {
            k: float(v)
            for k, v in data.items()
            if k.startswith(("input_", "target_", "output_")) and v not in ("-inf", "inf")
        }
    except (ValueError, TypeError):
        return None


def filter_chain(measured: dict[str, float] | None = None) -> str:
    """The mastering chain, as an ffmpeg -af argument.

    Order matters: clean up the spectrum, even out dynamics, set the loudness,
    then catch anything still over the peak ceiling.
    """
    stages = pre_filters()
    if measured:
        stages.append(
            "loudnorm=I={i}:TP={tp}:LRA={lra}:measured_I={mi}:measured_TP={mtp}:"
            "measured_LRA={mlra}:measured_thresh={mth}:offset={off}:linear=true".format(
                i=TARGET_LUFS,
                tp=TARGET_PEAK_DB,
                lra=TARGET_LRA,
                mi=measured.get("input_i", -26.0),
                mtp=measured.get("input_tp", -2.0),
                mlra=measured.get("input_lra", 4.0),
                mth=measured.get("input_thresh", -36.0),
                off=measured.get("target_offset", 0.0),
            )
        )
    else:
        # Single pass still gets the level roughly right.
        stages.append(f"loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK_DB}:LRA={TARGET_LRA}")
    stages.append(f"alimiter=limit={TARGET_PEAK_DB}dB:level=disabled")
    return ",".join(stages)


def verify(path: str | Path) -> dict[str, float]:
    """Measure a finished file against the ACX numbers."""
    result = ffmpeg.run(
        [
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "astats=metadata=1",
            "-f",
            "null",
            "-",
        ],
        text=True,
    )
    out: dict[str, float] = {}
    for key, label in (
        ("RMS level dB", "rms_db"),
        ("Peak level dB", "peak_db"),
        ("Noise floor dB", "noise_floor_db"),
    ):
        # The overall summary comes last, after any per-channel sections.
        values = re.findall(rf"{re.escape(key)}: (-?[\d.]+|-?inf)", result.stderr)
        if values and values[-1] not in ("-inf", "inf"):
            out[label] = float(values[-1])
    return out


def meets_acx(stats: dict[str, float]) -> bool:
    rms = stats.get("rms_db")
    peak = stats.get("peak_db")
    if rms is None or peak is None:
        return False
    low, high = ACX_RMS_RANGE_DB
    return low <= rms <= high and peak <= ACX_PEAK_DB
