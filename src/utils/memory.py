"""Utility functions for memory management and device monitoring."""

import functools
from typing import Any

import humanize
import jax


def show_hbm_usage() -> None:
    """Displays memory usage per device."""
    fmt_size = functools.partial(humanize.naturalsize, binary=True)

    for d in jax.local_devices():
        stats = d.memory_stats()
        if stats:
            used = stats["bytes_in_use"]
            limit = stats["bytes_limit"]
            print(
                f"Using {fmt_size(used)} / {fmt_size(limit)} "
                f"({used/limit:.1%}) on {d}"
            )


def format_memory_stats(stats: dict[str, Any]) -> str:
    """Format memory statistics for display.

    Args:
        stats: Dictionary of memory statistics

    Returns:
        Formatted string with memory usage information
    """
    if not stats:
        return "No memory stats available"

    fmt_size = functools.partial(humanize.naturalsize, binary=True)
    used = stats.get("bytes_in_use", 0)
    limit = stats.get("bytes_limit", 1)
    return (
        f"{fmt_size(used)} / {fmt_size(limit)} "
        f"({used/limit:.1%})"
    )
