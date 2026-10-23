import numpy as np
import pytest

from src.features.head_pose import (
    MODEL_POINTS,
    POSE_INDICES,
    estimate_head_pose,
    pose_from_mediapipe_matrix,
)

W, H = 640, 480


def rot_x(a):
    a = np.radians(a)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def rot_y(a):
    a = np.radians(a)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_z(a):
    a = np.radians(a)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def project(pitch, yaw, roll):
    R = rot_z(roll) @ rot_y(yaw) @ rot_x(pitch)
    cam = (R @ MODEL_POINTS.T).T + np.array([0, 0, 900.0])
    u = W * cam[:, 0] / cam[:, 2] + W / 2
    v = W * cam[:, 1] / cam[:, 2] + H / 2
    points = np.zeros((478, 2))
    points[POSE_INDICES] = np.stack([u, v], axis=1)
    return points, R


@pytest.mark.parametrize("pitch,yaw,roll", [(0, 0, 0), (15, 0, 0), (-10, 25, 0), (5, -30, 8)])
def test_solvepnp_recovers_known_angles(pitch, yaw, roll):
    points, _ = project(pitch, yaw, roll)
    result = estimate_head_pose(points, W, H)
    assert result == pytest.approx((pitch, yaw, roll), abs=0.5)


def test_mediapipe_matrix_conversion_matches():
    _, R_cv = project(12, -20, 6)
    flip = np.diag([1.0, -1.0, -1.0])
    matrix = np.eye(4)
    matrix[:3, :3] = 1.0 * (flip @ R_cv @ flip)
    assert pose_from_mediapipe_matrix(matrix) == pytest.approx((12, -20, 6), abs=0.01)