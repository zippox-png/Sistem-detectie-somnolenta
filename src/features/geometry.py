import numpy as np


def to_pixels(landmarks, width, height):
    return landmarks[:, :2] * np.array([width, height], dtype=np.float32)


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))