import os
import cv2
import argparse
from services.predictor import GesturePredictor
from adapters.hand_detector import HandDetector

def test_mudras(val_dir: str = "archive (2)/images/val", samples_per_mudra: int = 1, show_window: bool = False):
    """
    Evaluates trained mudra predictor across classes from validation set
    and prints predictions and confidence.
    """
    predictor = GesturePredictor("outputs/models/mudra_mlp")
    detector = HandDetector(static_image_mode=True)

    print("=" * 70)
    print(" RUNNING ARUNODAYA MUDRA PREDICTOR TEST")
    print("=" * 70)

    if not os.path.exists(val_dir):
        print(f"Validation directory not found: {val_dir}")
        return

    all_files = os.listdir(val_dir)
    sorted_classes = sorted(predictor.classes, key=lambda x: len(x), reverse=True)

    tested = 0
    correct = 0

    for c in predictor.classes:
        matched = [f for f in all_files if f.lower().startswith(c.lower())]
        if not matched:
            continue

        for f in matched[:samples_per_mudra]:
            img_path = os.path.join(val_dir, f)
            img = cv2.imread(img_path)
            if img is None:
                continue

            coords, annotated = detector.find_landmarks(img)
            pred, conf, top3 = predictor.predict(coords)

            is_match = (pred.lower() == c.lower())
            tested += 1
            if is_match:
                correct += 1

            status = "[MATCH]" if is_match else "[DIFF ]"
            print(f"{status} True: {c:<16} | Pred: {pred:<16} | Conf: {conf*100:5.1f}%")

            if show_window:
                cv2.putText(annotated, f"Pred: {pred} ({conf*100:.1f}%)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if is_match else (0, 0, 255), 2)
                cv2.imshow("Mudra Test Viewer", annotated)
                if cv2.waitKey(600) & 0xFF == ord('q'):
                    break

    detector.close()
    if show_window:
        cv2.destroyAllWindows()

    acc = (correct / tested * 100.0) if tested > 0 else 0.0
    print("=" * 70)
    print(f" Tested: {tested} images | Correct: {correct} | Accuracy: {acc:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1, help="Samples per mudra to test")
    parser.add_argument("--show", action="store_true", help="Display visual window for each image")
    args = parser.parse_args()
    test_mudras(samples_per_mudra=args.samples, show_window=args.show)
