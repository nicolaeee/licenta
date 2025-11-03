import cv2
from src.face_detector import FaceDetector
from src.eye_tracking import EyeTracker
from src.utils import draw_landmarks
from src.alerts import Alerts

def main():
    cap = cv2.VideoCapture(0)

    detector = FaceDetector()
    eyes = EyeTracker()
    alert = Alerts()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                left_eye, right_eye = eyes.get_eye_positions(frame, face_landmarks.landmark)

                draw_landmarks(frame, left_eye, (0, 255, 0))
                draw_landmarks(frame, right_eye, (255, 0, 0))

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
