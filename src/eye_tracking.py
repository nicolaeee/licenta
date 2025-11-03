from .utils import get_landmark_coords

class EyeTracker:
    def __init__(self):
        # Landmark indices for eyes based on MediaPipe FaceMesh
        self.left_eye_idx = [33, 133]
        self.right_eye_idx = [362, 263]

    def get_eye_positions(self, frame, landmarks):
        left_eye = get_landmark_coords(frame, landmarks, self.left_eye_idx)
        right_eye = get_landmark_coords(frame, landmarks, self.right_eye_idx)
        return left_eye, right_eye
