import os
import sys

# Add project root to Python path so we can import adapters, services, core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from adapters.hand_detector import HandDetector

def create_dataset_dirs(gesture_name: str, split: str = "train"):
    """Create folders for correct and wrong gesture images."""
    correct_dir = os.path.join("dataset", split, gesture_name, "correct")
    wrong_dir = os.path.join("dataset", split, gesture_name, "wrong")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)
    return correct_dir, wrong_dir

def capture_gestures(gesture_name: str, split: str = "train"):
    """
    Webcam capture helper to collect gesture images.
    Press 'c' to save a CORRECT image frame.
    Press 'w' to save a WRONG image frame.
    Press 'q' to quit.
    """
    correct_dir, wrong_dir = create_dataset_dirs(gesture_name, split)
    detector = HandDetector(max_num_hands=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print(f"--- Capturing dataset for gesture: '{gesture_name}' ({split}) ---")
    print("Press 'c' for CORRECT gesture")
    print("Press 'w' for WRONG gesture")
    print("Press 'q' to QUIT")

    correct_count = len(os.listdir(correct_dir))
    wrong_count = len(os.listdir(wrong_dir))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        coords, display_frame = detector.find_landmarks(frame)

        hand_status = "HAND DETECTED 🟩" if coords is not None else "NO HAND 🟥"

        # Display counts and status on window
        cv2.putText(display_frame, f"Gesture: {gesture_name} ({split})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Status: {hand_status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"[C]orrect: {correct_count} | [W]rong: {wrong_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gesture Data Collector", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            correct_count += 1
            filename = os.path.join(correct_dir, f"frame_{correct_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        elif key == ord('w'):
            wrong_count += 1
            filename = os.path.join(wrong_dir, f"frame_{wrong_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Done capturing images!")

if __name__ == "__main__":
    gesture = input("Enter gesture name (e.g. thumbs_up): ").strip() or "thumbs_up"
    split_choice = input("Enter split (train/val) [default: train]: ").strip() or "train"
    capture_gestures(gesture, split_choice)
