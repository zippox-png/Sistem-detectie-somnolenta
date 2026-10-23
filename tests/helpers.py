import numpy as np

from src.features.ear import LEFT_EYE, RIGHT_EYE


def empty_points():
    return np.zeros((478, 2), dtype=np.float64)


def set_eye(points, indices, x0, open_height):
    p1, p2, p3, p4, p5, p6 = indices
    points[p1] = (x0, 0)
    points[p4] = (x0 + 30, 0)
    points[p2] = (x0 + 10, -open_height / 2)
    points[p3] = (x0 + 20, -open_height / 2)
    points[p6] = (x0 + 10, open_height / 2)
    points[p5] = (x0 + 20, open_height / 2)


def make_face_with_eyes(open_height):
    points = empty_points()
    set_eye(points, RIGHT_EYE, 0, open_height)
    set_eye(points, LEFT_EYE, 100, open_height)
    return points