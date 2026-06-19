"""
evalgrid/progress.py: Friendly progress reporting for long evaluation runs.

Uses ``rich`` for a beautiful bar when available, falls back to a tiny stdlib
implementation otherwise. Never raises — progress bars must not break an eval.
"""

import sys
import time
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    description: str = "Evaluating",
    total: Optional[int] = None,
    quiet: bool = False,
) -> Iterator[T]:
    """
    Yield items from ``iterable`` while printing a progress bar.

    Args:
        iterable:    Any iterable; pass a list when you can so total is known.
        description: Label shown next to the bar.
        total:       Item count (auto-derived from len() when possible).
        quiet:       When True, no output is produced — items pass through unchanged.

    Yields:
        Each item from the original iterable, in order.
    """
    if quiet:
        yield from iterable
        return

    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    # Prefer rich if available — it gives the best UX
    try:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task(description, total=total)
            for item in iterable:
                yield item
                progress.advance(task_id)
        return
    except ImportError:
        pass

    # Fallback: simple stdlib progress bar (no extra dependencies required)
    yield from _stdlib_progress(iterable, description, total)


def _stdlib_progress(
    iterable: Iterable[T],
    description: str,
    total: Optional[int],
) -> Iterator[T]:
    """Tiny dependency-free progress bar — works everywhere."""
    start = time.time()
    count = 0
    bar_width = 30
    for item in iterable:
        yield item
        count += 1
        if total is not None and total > 0:
            ratio = count / total
            filled = int(bar_width * ratio)
            bar = "█" * filled + "░" * (bar_width - filled)
            elapsed = time.time() - start
            eta = (elapsed / count) * (total - count) if count > 0 else 0
            sys.stderr.write(
                f"\r{description}: [{bar}] {count}/{total} ({ratio:.0%}) — ETA {eta:.1f}s"
            )
        else:
            sys.stderr.write(f"\r{description}: {count} items processed")
        sys.stderr.flush()
    elapsed = time.time() - start
    sys.stderr.write(f"\r{description}: {count}/{total or count} done in {elapsed:.2f}s\n")
    sys.stderr.flush()
