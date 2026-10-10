import cv2
from services.predictor import GesturePredictor
from services.smoothing import PredictionSmoother
from adapters.hand_detector import HandDetector

def main():
    """
    Main Production Entry Point: Real-Time Multi-Hand Gesture Recognition.
    """
    # Lower confidence threshold slightly so active gestures are classified smoothly
    predictor = GesturePredictor(model_dir="outputs/models/landmark_mlp", confidence_threshold=0.50)
    smoother = PredictionSmoother(buffer_size=5, cooldown_seconds=0.2)
    detector = HandDetector(max_num_hands=2)

    # Use DirectShow backend on Windows for smooth, fast camera frame capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera feed.")
        return

    print("=====================================================")
    print(" [LIVE] Real-Time Mudra Recognition (Webcam)         ")
    print(" Press 'q' in the camera window to exit.             ")
    print("=====================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        all_coords, annotated_frame = detector.find_landmarks(frame)

        raw_label = "NO HAND DETECTED"
        confidence = 0.0
        top3 = []

        if all_coords is not None and len(all_coords) > 0:
            # Predict mudra using multi-hand keypoint adapter
            raw_label, confidence, top3 = predictor.predict(all_coords)

        # Smooth prediction across frames
        stable_label = smoother.update(raw_label)

        # Visual feedback colors
        if stable_label in ("NO HAND DETECTED", "UNCERTAIN"):
            color = (0, 215, 255) if stable_label == "UNCERTAIN" else (160, 160, 160)
        else:
            color = (0, 255, 120)  # Bright Green/Cyan for detected mudra

        # Display HUD overlay
        num_hands = len(all_coords) if all_coords else 0
        cv2.rectangle(annotated_frame, (10, 10), (480, 105), (20, 20, 20), -1)
        cv2.rectangle(annotated_frame, (10, 10), (480, 105), color, 2)
        cv2.putText(annotated_frame, f"Mudra: {stable_label}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        
        alt_text = f"Top-2: {top3[1][0]} ({top3[1][1]*100:.0f}%)" if len(top3) > 1 and stable_label != "NO HAND DETECTED" else ""
        cv2.putText(annotated_frame, f"Hands: {num_hands} | Conf: {confidence*100:.1f}%  {alt_text}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

        cv2.imshow("Arunodaya Mudra Recognition", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()
