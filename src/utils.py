import cv2

def get_landmark_coords(frame, landmarks, idx_list):
    h, w = frame.shape[:2]
    points = []
    for idx in idx_list:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append((x, y))
    return points

def draw_landmarks(frame, points, color=(0,255,0)):
    for pt in points:
        cv2.circle(frame, pt, 3, color, -1)