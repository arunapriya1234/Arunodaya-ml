import os
import sys

# Add project root to Python path so we can import adapters, services, core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from services.predictor import GesturePredictor
from adapters.hand_detector import HandDetector

def run_live_inference(model_dir: str = "outputs/models/landmark_mlp"):
    """
    Step 3: Real-Time OpenCV + MediaPipe Live Webcam Loop.
    """
    predictor = GesturePredictor(model_dir=model_dir, confidence_threshold=0.70)
    detector = HandDetector(static_image_mode=False, max_num_hands=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- Starting Real-Time Gesture Inference Loop ---")
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        coords, annotated_frame = detector.find_landmarks(frame)

        label = "NO HAND DETECTED"
        confidence = 0.0

        if coords is not None:
            label, confidence = predictor.predict(coords)

        # UI Color feedback based on prediction
        if label == "CORRECT":
            color = (0, 255, 0)      # Green
        elif label == "INCORRECT":
            color = (0, 0, 255)      # Red
        elif label == "UNCERTAIN":
            color = (0, 255, 255)    # Yellow
        else:
            color = (200, 200, 200)  # Gray

        # Overlay HUD text
        cv2.rectangle(annotated_frame, (10, 10), (380, 80), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Prediction: {label}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated_frame, f"Confidence: {confidence*100:.1f}%", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Live Hand Gesture Recognition", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    run_live_inference()
