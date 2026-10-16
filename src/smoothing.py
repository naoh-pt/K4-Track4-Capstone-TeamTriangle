"""Time-based hysteresis, independent of inference FPS."""
import math


class SeatState:
    def __init__(self, on_seconds=0.5, off_seconds=1.0, max_gap_seconds=0.5):
        for value in (on_seconds, off_seconds, max_gap_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Durations must be positive and finite")
        self.on_seconds = on_seconds
        self.off_seconds = off_seconds
        self.max_gap_seconds = max_gap_seconds
        self.occupied = None  # unknown until enough evidence has arrived
        self._pending = None
        self._since = None
        self._last = None

    def update(self, detected, timestamp):
        """Use monotonic seconds; None means missing evidence, never empty."""
        if detected is not None and type(detected) is not bool:
            raise ValueError("Detection must be bool or None")
        if not math.isfinite(timestamp) or (self._last is not None and timestamp < self._last):
            raise ValueError("Timestamps must be finite and non-decreasing")
        if self._last is not None and timestamp - self._last > self.max_gap_seconds:
            self._pending = self._since = None
        self._last = timestamp
        if detected is None or detected == self.occupied:
            self._pending = self._since = None
            return self.occupied
        if detected != self._pending:
            self._pending, self._since = detected, timestamp
        required = self.on_seconds if detected else self.off_seconds
        if timestamp - self._since >= required:
            self.occupied = detected
            self._pending = self._since = None
        return self.occupied
