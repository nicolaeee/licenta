import math
from .utils import get_landmark_coords

class EyeTracker:
    def __init__(self):
        # Eye landmarks for EAR
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [362, 385, 387, 263, 373, 380]

    def euclidean_dist(self, p1, p2):
        return math.dist(p1, p2)

    def EAR(self, eye_points):
        p1, p2, p3, p4, p5, p6 = eye_points
        vertical1 = self.euclidean_dist(p2, p6)
        vertical2 = self.euclidean_dist(p3, p5)
        horizontal = self.euclidean_dist(p1, p4)
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    def get_eye_landmarks(self, frame, landmarks, eye_idx):
        return get_landmark_coords(frame, landmarks, eye_idx)

    def get_ear_values(self, frame, face_landmarks):
        left_eye_pts = self.get_eye_landmarks(frame, face_landmarks, self.left_eye_idx)
        right_eye_pts = self.get_eye_landmarks(frame, face_landmarks, self.right_eye_idx)

        left_ear = self.EAR(left_eye_pts)
        right_ear = self.EAR(right_eye_pts)
        return (left_ear + right_ear) / 2.0, left_eye_pts, right_eye_pts
