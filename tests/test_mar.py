import pytest

from src.features.mar import compute_mar
from tests.helpers import empty_points


def make_mouth(opening):
    points = empty_points()
    points[78] = (-20, 0)
    points[308] = (20, 0)
    points[13] = (0, -opening / 2)
    points[14] = (0, opening / 2)
    return points


def test_closed_mouth_has_zero_mar():
    assert compute_mar(make_mouth(0)) == pytest.approx(0.0)


def test_open_mouth_has_high_mar():
    assert compute_mar(make_mouth(20)) == pytest.approx(0.5)


def test_mar_grows_with_opening():
    assert compute_mar(make_mouth(30)) > compute_mar(make_mouth(10))