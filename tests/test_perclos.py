import pytest

from src.features.perclos import Perclos


def run(p, pattern, duration_s, hz=10.0, start=0.0):
    n = int(duration_s * hz)
    value = 0.0
    for i in range(n):
        t = start + i / hz
        value = p.update(pattern(t), t)
    return value


def test_always_open_is_zero():
    assert run(Perclos(), lambda t: False, 30) == pytest.approx(0.0)


def test_always_closed_is_one():
    assert run(Perclos(), lambda t: True, 30) == pytest.approx(1.0, abs=0.02)


def test_half_the_time_closed():
    value = run(Perclos(), lambda t: (t % 2) < 1, 120)
    assert value == pytest.approx(0.5, abs=0.03)


def test_uses_real_time_not_frame_count():
    slow = run(Perclos(), lambda t: t % 2 < 1, 120, hz=5.0)
    fast = run(Perclos(), lambda t: t % 2 < 1, 120, hz=30.0)
    assert slow == pytest.approx(fast, abs=0.03)


def test_old_data_leaves_the_window():
    p = Perclos(window_s=60.0)
    run(p, lambda t: True, 60, start=0.0)
    value = run(p, lambda t: False, 70, start=60.0)
    assert value == pytest.approx(0.0, abs=0.02)