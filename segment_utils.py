"""Shared reasoning-segment splitter used by attribution and SFT."""

import re


CUE_PATTERN = re.compile(
    r"(\n\nWait|\n\nAlternatively|\n\nBut wait|\n\nBut alternatively|"
    r"\n\nBut just to|\n\nHowever|\n\nNot sure|\n\nGoing back|"
    r"\n\nBacktrack|\n\nTrace back|\n\nAnother)"
)

SEGMENT_MODES = ("cue", "paragraph")


def split_segments(text: str, mode: str = "paragraph") -> list[str]:
    """Split *text* without losing delimiters or changing its contents."""
    if mode not in SEGMENT_MODES:
        raise ValueError(f"Unknown segment mode: {mode}")
    if not text:
        return []

    pattern = CUE_PATTERN if mode == "cue" else re.compile(r"(\n\n)")
    parts = pattern.split(text)
    segments = [parts[0]]
    for index in range(1, len(parts), 2):
        tail = parts[index + 1] if index + 1 < len(parts) else ""
        segments.append(parts[index] + tail)

    segments = [segment for segment in segments if segment]
    if "".join(segments) != text:
        raise RuntimeError("Segment split changed the source text")
    return segments
