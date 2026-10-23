import pytest

from src.features.ear import compute_ear
from tests.helpers import make_face_with_eyes


def test_open_eye_has_high_ear():
    ear, left, right = compute_ear(make_face_with_eyes(open_height=10))
    assert ear == pytest.approx(10 / 30)
    assert left == pytest.approx(right)


def test_closed_eye_has_zero_ear():
    ear, _, _ = compute_ear(make_face_with_eyes(open_height=0))
    assert ear == pytest.approx(0.0)


def test_ear_decreases_when_eye_closes():
    open_ear, _, _ = compute_ear(make_face_with_eyes(10))
    half_ear, _, _ = compute_ear(make_face_with_eyes(5))
    assert open_ear > half_ear > 0


def test_ear_does_not_depend_on_scale():
    small = make_face_with_eyes(10)
    big = small * 7
    assert compute_ear(small)[0] == pytest.approx(compute_ear(big)[0])