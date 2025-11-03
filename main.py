import cv2
import numpy as np
import mediapipe as mp

# Inițializare MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Landmark-uri ochi (simplificat)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def eye_ratio(landmarks, eye, img_w, img_h):
    pts = [(int(landmarks[i].x*img_w), int(landmarks[i].y*img_h)) for i in eye]

    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    ratio = (A + B) / (2.0 * C + 1e-6)
    return ratio

# Deschidere camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        left_ratio = eye_ratio(lm, LEFT_EYE, w, h)
        right_ratio = eye_ratio(lm, RIGHT_EYE, w, h)
        ratio = (left_ratio + right_ratio) / 2

        # Prag detectare somnolență (tune-abil)
        if ratio < 0.18:
            cv2.putText(frame, "DROWSY!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "Awake", (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 3)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
