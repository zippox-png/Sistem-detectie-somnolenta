from src.features.geometry import distance

UPPER_LIP = 13
LOWER_LIP = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


def compute_mar(points):
    width = distance(points[MOUTH_LEFT], points[MOUTH_RIGHT])
    if width == 0:
        return 0.0
    return distance(points[UPPER_LIP], points[LOWER_LIP]) / width