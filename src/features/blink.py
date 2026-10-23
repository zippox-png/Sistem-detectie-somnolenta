from collections import deque


class BlinkDetector:
    def __init__(self, ear_threshold=0.21, min_frames=2, window_s=60.0):
        self.ear_threshold = ear_threshold
        self.min_frames = min_frames
        self.window_s = window_s
        self.blinks = deque()
        self._closed_frames = 0
        self._closed_start = None
        self._first_ts = None
        self._last_ts = 0.0

    def update(self, ear, timestamp):
        if self._first_ts is None:
            self._first_ts = timestamp
        self._last_ts = timestamp
        finished_blink = None

        if ear is None:
            self._reset_closure()
        elif ear < self.ear_threshold:
            if self._closed_frames == 0:
                self._closed_start = timestamp
            self._closed_frames += 1
        else:
            if self._closed_frames >= self.min_frames:
                duration = timestamp - self._closed_start
                self.blinks.append((timestamp, duration))
                finished_blink = duration
            self._reset_closure()

        self._purge(timestamp)
        return finished_blink

    def _reset_closure(self):
        self._closed_frames = 0
        self._closed_start = None

    def _purge(self, now):
        while self.blinks and self.blinks[0][0] < now - self.window_s:
            self.blinks.popleft()

    @property
    def closed_duration(self):
        if self._closed_start is None:
            return 0.0
        return self._last_ts - self._closed_start

    @property
    def blink_rate(self):
        if self._first_ts is None:
            return 0.0
        effective_window = max(min(self.window_s, self._last_ts - self._first_ts), 10.0)
        return len(self.blinks) * 60.0 / effective_window

    @property
    def mean_duration(self):
        if not self.blinks:
            return 0.0
        return sum(d for _, d in self.blinks) / len(self.blinks)