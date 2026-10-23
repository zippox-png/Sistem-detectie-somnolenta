from collections import deque


class Perclos:
    def __init__(self, window_s=60.0, max_gap_s=0.5):
        self.window_s = window_s
        self.max_gap_s = max_gap_s
        self._intervals = deque()
        self._closed_time = 0.0
        self._total_time = 0.0
        self._prev_ts = None
        self._prev_closed = False

    def update(self, closed, timestamp):
        if self._prev_ts is not None:
            dt = min(timestamp - self._prev_ts, self.max_gap_s)
            if dt > 0:
                self._intervals.append((timestamp, dt, self._prev_closed))
                self._total_time += dt
                if self._prev_closed:
                    self._closed_time += dt
        self._prev_ts = timestamp
        self._prev_closed = bool(closed)

        limit = timestamp - self.window_s
        while self._intervals and self._intervals[0][0] < limit:
            _, dt, was_closed = self._intervals.popleft()
            self._total_time -= dt
            if was_closed:
                self._closed_time -= dt

        return self.value

    @property
    def value(self):
        if self._total_time <= 0:
            return 0.0
        return max(0.0, min(1.0, self._closed_time / self._total_time))