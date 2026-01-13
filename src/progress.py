"""
Progress reporting interface for RLM operations.

Implements: Phase 4 Massive Context (SPEC-01.05)

Provides progress callbacks, cancellation support, and
streaming progress updates during long operations.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable


class OperationType(Enum):
    """Types of operations that can be tracked."""

    INDEXING = "indexing"
    SEARCHING = "searching"
    CHUNKING = "chunking"
    ANALYZING = "analyzing"
    DECOMPOSING = "decomposing"
    SYNTHESIZING = "synthesizing"
    LLM_CALL = "llm_call"
    BATCH_OPERATION = "batch_operation"
    MAP_REDUCE = "map_reduce"
    CONTEXT_LOAD = "context_load"


@dataclass
class ProgressUpdate:
    """
    Progress update event.

    Provides information about current operation status.
    """

    operation: OperationType
    current: int
    total: int
    message: str
    elapsed_ms: float = 0.0
    estimated_remaining_ms: float | None = None
    detail: str = ""
    depth: int = 0  # Nesting depth for recursive operations


@dataclass
class ProgressStats:
    """Statistics about operation progress."""

    total_operations: int = 0
    completed_operations: int = 0
    failed_operations: int = 0
    total_time_ms: float = 0.0
    operations_per_second: float = 0.0


@runtime_checkable
class ProgressCallback(Protocol):
    """
    Protocol for progress callbacks.

    Implement this to receive progress updates during operations.
    """

    def on_progress(self, update: ProgressUpdate) -> None:
        """
        Called when progress is made.

        Args:
            update: Progress update information
        """
        ...

    def on_complete(self, stats: ProgressStats) -> None:
        """
        Called when operation completes.

        Args:
            stats: Final statistics
        """
        ...

    def on_error(self, error: Exception, update: ProgressUpdate) -> None:
        """
        Called when an error occurs.

        Args:
            error: The exception that occurred
            update: Progress state when error occurred
        """
        ...


class CancellationToken:
    """
    Token for checking and requesting cancellation.

    Thread-safe and async-safe.
    """

    def __init__(self) -> None:
        """Initialize cancellation token."""
        self._cancelled = False
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled

    def check(self) -> None:
        """
        Check if cancelled and raise if so.

        Raises:
            CancelledException: If cancellation was requested
        """
        if self._cancelled:
            raise CancelledException("Operation was cancelled")

    async def wait_for_cancellation(self, timeout: float | None = None) -> bool:
        """
        Wait for cancellation.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if cancelled, False if timeout
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
            return True
        except TimeoutError:
            return False


class CancelledException(Exception):
    """Raised when an operation is cancelled."""

    pass


@dataclass
class ProgressContext:
    """
    Context manager for tracking progress.

    Provides progress reporting, timing, and cancellation.
    """

    operation: OperationType
    total: int
    callback: ProgressCallback | None = None
    cancellation_token: CancellationToken | None = None
    message_template: str = "{operation}: {current}/{total}"
    detail: str = ""
    depth: int = 0

    # Internal state
    _current: int = field(default=0, init=False)
    _start_time: float = field(default=0.0, init=False)
    _stats: ProgressStats = field(default_factory=ProgressStats, init=False)

    def __post_init__(self) -> None:
        """Initialize progress tracking."""
        self._stats.total_operations = self.total

    def __enter__(self) -> ProgressContext:
        """Start progress tracking."""
        self._start_time = time.perf_counter()
        self._report_progress()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        """Complete progress tracking."""
        elapsed = (time.perf_counter() - self._start_time) * 1000
        self._stats.total_time_ms = elapsed
        if self._stats.total_time_ms > 0:
            self._stats.operations_per_second = self._stats.completed_operations / (
                self._stats.total_time_ms / 1000
            )

        if exc_val is not None:
            self._stats.failed_operations += 1
            if self.callback:
                self.callback.on_error(exc_val, self._create_update())
        elif self.callback:
            self.callback.on_complete(self._stats)

    async def __aenter__(self) -> ProgressContext:
        """Async start progress tracking."""
        return self.__enter__()

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: object
    ) -> None:
        """Async complete progress tracking."""
        self.__exit__(exc_type, exc_val, exc_tb)

    def advance(self, count: int = 1, detail: str = "") -> None:
        """
        Advance progress by count.

        Args:
            count: Number of steps to advance
            detail: Optional detail message

        Raises:
            CancelledException: If cancelled
        """
        if self.cancellation_token:
            self.cancellation_token.check()

        self._current += count
        self._stats.completed_operations += count
        if detail:
            self.detail = detail
        self._report_progress()

    def set_progress(self, current: int, detail: str = "") -> None:
        """
        Set absolute progress.

        Args:
            current: Current position
            detail: Optional detail message

        Raises:
            CancelledException: If cancelled
        """
        if self.cancellation_token:
            self.cancellation_token.check()

        self._stats.completed_operations += current - self._current
        self._current = current
        if detail:
            self.detail = detail
        self._report_progress()

    def _create_update(self) -> ProgressUpdate:
        """Create a progress update."""
        elapsed = (time.perf_counter() - self._start_time) * 1000

        # Estimate remaining time
        estimated_remaining: float | None = None
        if self._current > 0 and self._current < self.total:
            rate = self._current / elapsed
            remaining = self.total - self._current
            estimated_remaining = remaining / rate if rate > 0 else None

        message = self.message_template.format(
            operation=self.operation.value,
            current=self._current,
            total=self.total,
        )

        return ProgressUpdate(
            operation=self.operation,
            current=self._current,
            total=self.total,
            message=message,
            elapsed_ms=elapsed,
            estimated_remaining_ms=estimated_remaining,
            detail=self.detail,
            depth=self.depth,
        )

    def _report_progress(self) -> None:
        """Report progress to callback."""
        if self.callback:
            self.callback.on_progress(self._create_update())


class ConsoleProgressCallback:
    """
    Console-based progress callback.

    Prints progress updates to stdout in a user-friendly format.
    """

    def __init__(
        self,
        show_elapsed: bool = True,
        show_eta: bool = True,
        prefix: str = "",
    ):
        """
        Initialize console callback.

        Args:
            show_elapsed: Show elapsed time
            show_eta: Show estimated time remaining
            prefix: Prefix for all messages
        """
        self.show_elapsed = show_elapsed
        self.show_eta = show_eta
        self.prefix = prefix

    def on_progress(self, update: ProgressUpdate) -> None:
        """Print progress update."""
        parts = [self.prefix] if self.prefix else []

        # Main progress message
        indent = "  " * update.depth
        parts.append(f"{indent}{update.message}")

        # Percentage
        if update.total > 0:
            pct = (update.current / update.total) * 100
            parts.append(f"({pct:.1f}%)")

        # Elapsed time
        if self.show_elapsed and update.elapsed_ms > 0:
            if update.elapsed_ms < 1000:
                parts.append(f"[{update.elapsed_ms:.0f}ms]")
            else:
                parts.append(f"[{update.elapsed_ms / 1000:.1f}s]")

        # ETA
        if self.show_eta and update.estimated_remaining_ms:
            if update.estimated_remaining_ms < 1000:
                parts.append(f"~{update.estimated_remaining_ms:.0f}ms left")
            else:
                parts.append(f"~{update.estimated_remaining_ms / 1000:.1f}s left")

        # Detail
        if update.detail:
            parts.append(f"- {update.detail}")

        print(" ".join(parts), flush=True)

    def on_complete(self, stats: ProgressStats) -> None:
        """Print completion message."""
        msg = f"{self.prefix}Completed: {stats.completed_operations}/{stats.total_operations}"
        if stats.total_time_ms > 0:
            if stats.total_time_ms < 1000:
                msg += f" in {stats.total_time_ms:.0f}ms"
            else:
                msg += f" in {stats.total_time_ms / 1000:.1f}s"
        if stats.operations_per_second > 0:
            msg += f" ({stats.operations_per_second:.1f} ops/sec)"
        print(msg, flush=True)

    def on_error(self, error: Exception, update: ProgressUpdate) -> None:
        """Print error message."""
        print(f"{self.prefix}Error at {update.current}/{update.total}: {error}", flush=True)


class NullProgressCallback:
    """No-op progress callback for when progress reporting is disabled."""

    def on_progress(self, update: ProgressUpdate) -> None:
        """Ignore progress updates."""
        pass

    def on_complete(self, stats: ProgressStats) -> None:
        """Ignore completion."""
        pass

    def on_error(self, error: Exception, update: ProgressUpdate) -> None:
        """Ignore errors."""
        pass


class CompositeProgressCallback:
    """
    Combines multiple progress callbacks.

    Forwards all events to all registered callbacks.
    """

    def __init__(self, callbacks: list[ProgressCallback] | None = None):
        """
        Initialize composite callback.

        Args:
            callbacks: Initial list of callbacks
        """
        self._callbacks: list[ProgressCallback] = callbacks or []

    def add(self, callback: ProgressCallback) -> None:
        """Add a callback."""
        self._callbacks.append(callback)

    def remove(self, callback: ProgressCallback) -> None:
        """Remove a callback."""
        self._callbacks.remove(callback)

    def on_progress(self, update: ProgressUpdate) -> None:
        """Forward to all callbacks."""
        for cb in self._callbacks:
            cb.on_progress(update)

    def on_complete(self, stats: ProgressStats) -> None:
        """Forward to all callbacks."""
        for cb in self._callbacks:
            cb.on_complete(stats)

    def on_error(self, error: Exception, update: ProgressUpdate) -> None:
        """Forward to all callbacks."""
        for cb in self._callbacks:
            cb.on_error(error, update)


class ThrottledProgressCallback:
    """
    Throttles progress updates to avoid overwhelming output.

    Only forwards updates at most every `interval_ms` milliseconds.
    """

    def __init__(
        self,
        callback: ProgressCallback,
        interval_ms: float = 100,
        always_report_complete: bool = True,
    ):
        """
        Initialize throttled callback.

        Args:
            callback: Underlying callback to forward to
            interval_ms: Minimum time between updates
            always_report_complete: Always report when current == total
        """
        self._callback = callback
        self._interval_ms = interval_ms
        self._always_report_complete = always_report_complete
        self._last_report_time: float = 0

    def on_progress(self, update: ProgressUpdate) -> None:
        """Forward if enough time has passed."""
        now = time.perf_counter() * 1000

        # Always report completion
        if self._always_report_complete and update.current >= update.total:
            self._callback.on_progress(update)
            self._last_report_time = now
            return

        # Check throttle
        if now - self._last_report_time >= self._interval_ms:
            self._callback.on_progress(update)
            self._last_report_time = now

    def on_complete(self, stats: ProgressStats) -> None:
        """Always forward completion."""
        self._callback.on_complete(stats)

    def on_error(self, error: Exception, update: ProgressUpdate) -> None:
        """Always forward errors."""
        self._callback.on_error(error, update)


def create_progress_context(
    operation: OperationType,
    total: int,
    callback: ProgressCallback | Callable[[ProgressUpdate], None] | None = None,
    cancellation_token: CancellationToken | None = None,
    message_template: str | None = None,
    depth: int = 0,
) -> ProgressContext:
    """
    Create a progress context with convenient defaults.

    Args:
        operation: Type of operation
        total: Total number of steps
        callback: Progress callback (can be a simple function)
        cancellation_token: Optional cancellation token
        message_template: Custom message template
        depth: Nesting depth

    Returns:
        ProgressContext ready for use
    """
    # Wrap simple function in callback
    actual_callback: ProgressCallback | None = None
    if callback is not None:
        if isinstance(callback, ProgressCallback):
            actual_callback = callback
        else:
            # Wrap simple function
            class FunctionCallback:
                def __init__(self, fn: Callable[[ProgressUpdate], None]):
                    self._fn = fn

                def on_progress(self, update: ProgressUpdate) -> None:
                    self._fn(update)

                def on_complete(self, stats: ProgressStats) -> None:
                    pass

                def on_error(self, error: Exception, update: ProgressUpdate) -> None:
                    pass

            actual_callback = FunctionCallback(callback)

    template = message_template or "{operation}: {current}/{total}"

    return ProgressContext(
        operation=operation,
        total=total,
        callback=actual_callback,
        cancellation_token=cancellation_token,
        message_template=template,
        depth=depth,
    )


__all__ = [
    # Core types
    "OperationType",
    "ProgressUpdate",
    "ProgressStats",
    "ProgressCallback",
    # Cancellation
    "CancellationToken",
    "CancelledException",
    # Progress tracking
    "ProgressContext",
    "create_progress_context",
    # Callbacks
    "ConsoleProgressCallback",
    "NullProgressCallback",
    "CompositeProgressCallback",
    "ThrottledProgressCallback",
]
