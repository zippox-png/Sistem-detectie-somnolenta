import cv2
import numpy as np

POSE_INDICES = [1, 152, 33, 263, 61, 291]

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, 330.0, 65.0),
        (-225.0, -170.0, 135.0),
        (225.0, -170.0, 135.0),
        (-150.0, 150.0, 125.0),
        (150.0, 150.0, 125.0),
    ],
    dtype=np.float64,
)


def rotation_to_euler(R):
    yaw = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(pitch), float(yaw), float(roll)


def estimate_head_pose(points, width, height):
    image_points = np.asarray(points[POSE_INDICES], dtype=np.float64)
    camera_matrix = np.array(
        [[width, 0, width / 2.0], [0, width, height / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )
    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return rotation_to_euler(R)


def pose_from_mediapipe_matrix(matrix):
    R = np.asarray(matrix, dtype=np.float64)[:3, :3]
    R = R / np.linalg.norm(R, axis=0)
    flip = np.diag([1.0, -1.0, -1.0])
    return rotation_to_euler(flip @ R @ flip)