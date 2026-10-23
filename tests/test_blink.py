import pytest

from src.features.blink import BlinkDetector


def feed(detector, ears, fps=30.0, start=0.0):
    events = []
    for i, ear in enumerate(ears):
        events.append(detector.update(ear, start + i / fps))
    return events


def test_two_closed_frames_make_a_blink():
    d = BlinkDetector(ear_threshold=0.21, min_frames=2)
    events = feed(d, [0.3, 0.3, 0.1, 0.1, 0.3, 0.3])
    assert len([e for e in events if e is not None]) == 1
    assert d.mean_duration == pytest.approx(2 / 30)


def test_single_closed_frame_is_noise():
    d = BlinkDetector(min_frames=2)
    events = feed(d, [0.3, 0.1, 0.3, 0.3])
    assert all(e is None for e in events)
    assert len(d.blinks) == 0


def test_long_closure_reports_long_duration():
    d = BlinkDetector(min_frames=2)
    feed(d, [0.3] + [0.05] * 60 + [0.3])
    assert d.mean_duration > 1.5


def test_closed_duration_while_eyes_are_closed():
    d = BlinkDetector()
    feed(d, [0.3] + [0.05] * 31)
    assert d.closed_duration == pytest.approx(30 / 30, abs=0.05)


def test_old_blinks_leave_the_window():
    d = BlinkDetector(window_s=60.0)
    feed(d, [0.3, 0.1, 0.1, 0.3], start=0.0)
    assert len(d.blinks) == 1
    d.update(0.3, 100.0)
    assert len(d.blinks) == 0


def test_face_lost_interrupts_closure():
    d = BlinkDetector(min_frames=2)
    feed(d, [0.1, 0.1, None, 0.3, 0.3])
    assert len(d.blinks) == 0