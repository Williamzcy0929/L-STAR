"""Sampling-level controls for post-consensus domain annotation."""

from __future__ import annotations

SUPPORTED_SAMPLING_LEVELS = frozenset({"spot", "cell"})


def resolve_sampling_level(sampling_level: str) -> str:
    """Validate and normalize the only user-supplied sampling declaration."""
    normalized_level = str(sampling_level).strip().lower()
    if normalized_level not in SUPPORTED_SAMPLING_LEVELS:
        raise ValueError(
            "sampling_level must be one of: {}".format(
                ", ".join(sorted(SUPPORTED_SAMPLING_LEVELS))
            )
        )
    return normalized_level


__all__ = [
    "SUPPORTED_SAMPLING_LEVELS",
    "resolve_sampling_level",
]
