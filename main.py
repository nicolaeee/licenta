import cv2
import time
from src.face_detector import FaceDetector
from src.eye_tracking import EyeTracker
from src.fatigue_model import FatigueModel
from src.utils import draw_landmarks
from src.alerts import AlertSystem

EAR_THRESHOLD = 0.21
SLEEP_TIME = 2  # seconds

def main():
    cap = cv2.VideoCapture(0)

    detector = FaceDetector()
    eyes = EyeTracker()
    fatigue = FatigueModel()
    alert_system = AlertSystem()   # ✅ corect

    eye_closed_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                ear, left_eye_pts, right_eye_pts = eyes.get_ear_values(frame, face_landmarks.landmark)

                draw_landmarks(frame, left_eye_pts, (0, 255, 0))
                draw_landmarks(frame, right_eye_pts, (0, 255, 0))

                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                if ear < EAR_THRESHOLD:
                    if eye_closed_start is None:
                        eye_closed_start = time.time()

                    elapsed = time.time() - eye_closed_start

                    if elapsed >= SLEEP_TIME:
                        cv2.putText(frame, "DROWSINESS ALERT!", (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                        alert_system.trigger_alert()   # ✅ sunet si alerta
                else:
                    eye_closed_start = None
                    alert_system.reset_alert()       # ✅ dacă a revenit la normal

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
