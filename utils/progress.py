from __future__ import annotations

import math
import time
from typing import Callable, Optional


def format_duration(seconds: float) -> str:
    total_seconds = int(max(seconds, 0.0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


class TotalProgressReporter:
    """Print newline-delimited total progress at fixed percentage increments.

    ETA samples only work completed after this reporter was created.  This is
    important for resumed jobs: checkpointed work contributes to the displayed
    percentage, but not to the speed estimate for the new process.
    """

    def __init__(
        self,
        label: str,
        total_units: int,
        *,
        initial_completed: int = 0,
        resolution_percent: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if total_units <= 0:
            raise ValueError("total_units must be positive")
        if resolution_percent <= 0:
            raise ValueError("resolution_percent must be positive")

        self.label = label
        self.total_units = total_units
        self.initial_completed = min(max(initial_completed, 0), total_units)
        self.resolution_percent = resolution_percent
        self._buckets_per_percent = round(1.0 / resolution_percent)
        self._max_bucket = 100 * self._buckets_per_percent
        self._clock = clock
        self._started_at = clock()
        self._last_bucket: Optional[int] = None

    def _bucket_for(self, completed: int) -> int:
        ratio = min(max(completed, 0), self.total_units) / self.total_units
        return min(int(math.floor(ratio * self._max_bucket + 1e-9)), self._max_bucket)

    def _eta_text(self, bucket: int, observed_completed: int) -> str:
        completed_this_run = observed_completed - self.initial_completed
        if bucket >= self._max_bucket:
            return "00h00m00s"
        if completed_this_run <= 0:
            return "recalculating"
        elapsed = max(self._clock() - self._started_at, 1e-9)
        units_per_second = completed_this_run / elapsed
        displayed_completed = self.total_units * bucket / self._max_bucket
        return format_duration((self.total_units - displayed_completed) / units_per_second)

    def _print_bucket(self, bucket: int, completed: int) -> None:
        percent = bucket / self._buckets_per_percent
        elapsed = format_duration(self._clock() - self._started_at)
        print(
            f"{self.label} progress: {percent:5.1f}% | "
            f"elapsed_this_run {elapsed} | ETA {self._eta_text(bucket, completed)}",
            flush=True,
        )

    def start(self) -> None:
        bucket = self._bucket_for(self.initial_completed)
        self._last_bucket = bucket
        self._print_bucket(bucket, self.initial_completed)

    def update(self, completed: int) -> None:
        completed = min(max(completed, 0), self.total_units)
        bucket = self._bucket_for(completed)
        if self._last_bucket is None:
            self.start()
        assert self._last_bucket is not None
        if bucket <= self._last_bucket:
            return
        for crossed_bucket in range(self._last_bucket + 1, bucket + 1):
            self._print_bucket(crossed_bucket, completed)
        self._last_bucket = bucket

    def finish(self) -> None:
        self.update(self.total_units)
