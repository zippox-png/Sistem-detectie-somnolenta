from src.features.geometry import distance

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(eye):
    vertical_1 = distance(eye[1], eye[5])
    vertical_2 = distance(eye[2], eye[4])
    horizontal = distance(eye[0], eye[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_ear(points):
    left = eye_aspect_ratio(points[LEFT_EYE])
    right = eye_aspect_ratio(points[RIGHT_EYE])
    return (left + right) / 2.0, left, right