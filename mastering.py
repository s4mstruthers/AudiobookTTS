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
import re
import subprocess
from pathlib import Path

# Mid-ACX, and the usual target for spoken-word audiobooks.
TARGET_LUFS = -19.0
ACX_PEAK_DB = -3.0
# Aim below the ceiling: limiting exactly at -3.0 lands around -2.9993, which
# fails the spec by a whisker.
TARGET_PEAK_DB = -3.3
TARGET_LRA = 7.0

# Rumble and DC offset carry no speech but eat headroom.
HIGHPASS_HZ = 65
# Gentle: enough to tame TTS sibilance without lisping the output.
DEESSER_INTENSITY = 0.12


class MasteringError(RuntimeError):
    pass


def atempo_chain(tempo: float) -> str:
    """atempo only accepts 0.5–2.0 per stage, so chain it for larger shifts."""
    stages = []
    remaining = float(tempo)
    while remaining < 0.5:
        stages.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        stages.append("atempo=2.0")
        remaining /= 2.0
    stages.append(f"atempo={remaining:.6f}")
    return ",".join(stages)


def apply_filters(audio, sample_rate: int, filters: str):
    """Run an ffmpeg filter chain over an in-memory float32 array."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not filters:
        return audio
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-af", filters,
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "pipe:1",
        ],
        input=audio.tobytes(), capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        # Filtering is an enhancement; never lose the narration over it.
        return audio
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def measure(
    path: str | Path, input_args: list[str] | None = None
) -> dict[str, float] | None:
    """First pass: measure loudness. None if ffmpeg gives nothing usable.

    input_args lets the caller pass demuxer flags, e.g. the concat list the
    packager assembles rather than a plain audio file.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats",
            *(input_args or []), "-i", str(path),
            "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK_DB}:"
                   f"LRA={TARGET_LRA}:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return {k: float(v) for k, v in data.items()
                if k.startswith(("input_", "target_", "output_"))
                and v not in ("-inf", "inf")}
    except (ValueError, TypeError):
        return None


def filter_chain(measured: dict[str, float] | None = None) -> str:
    """The mastering chain, as an ffmpeg -af argument.

    Order matters: clean up the spectrum, even out dynamics, set the loudness,
    then catch anything still over the peak ceiling.
    """
    stages = [
        f"highpass=f={HIGHPASS_HZ}",
        f"deesser=i={DEESSER_INTENSITY}",
        # Light 2:1 — TTS is already consistent (measured 2-4 LU range), so
        # this tightens rather than squashes.
        "acompressor=threshold=-18dB:ratio=2:attack=15:release=250:makeup=1",
    ]

    if measured:
        stages.append(
            "loudnorm=I={i}:TP={tp}:LRA={lra}:measured_I={mi}:measured_TP={mtp}:"
            "measured_LRA={mlra}:measured_thresh={mth}:offset={off}:linear=true"
            .format(
                i=TARGET_LUFS, tp=TARGET_PEAK_DB, lra=TARGET_LRA,
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
    """Measure the finished file against the ACX numbers."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "astats=metadata=1", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    out: dict[str, float] = {}
    for key, label in (
        ("RMS level dB", "rms_db"),
        ("Peak level dB", "peak_db"),
        ("Noise floor dB", "noise_floor_db"),
    ):
        m = re.search(rf"{re.escape(key)}: (-?[\d.]+|-?inf)", result.stderr)
        if m and m.group(1) not in ("-inf", "inf"):
            out[label] = float(m.group(1))
    return out


def meets_acx(stats: dict[str, float]) -> bool:
    rms = stats.get("rms_db")
    peak = stats.get("peak_db")
    if rms is None or peak is None:
        return False
    return -23.0 <= rms <= -18.0 and peak <= ACX_PEAK_DB
