from __future__ import annotations

import cv2
import mediapipe as mp

from smart_gestures.alphabet.asl_model.model import ASLModel


def run_live_demo() -> None:
    model = ASLModel()
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("No webcam found. Connect a camera and try again.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                sample = [{"x": p.x, "y": p.y, "z": p.z} for p in hand_landmarks.landmark]
                label, confidence = model.predict(sample)
                text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("ASL Sign Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_demo()
