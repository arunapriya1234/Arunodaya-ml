import os
import csv
import numpy as np

def generate_pairs_csv(dataset_dir: str = "dataset", output_csv: str = "pairs.csv"):
    """
    Scans dataset folder and creates pairs metadata file pairs.csv.
    Label mapping:
    - correct -> 1
    - wrong / incorrect -> 0
    """
    rows = []
    splits = ["train", "val"]

    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        if not os.path.exists(split_path):
            continue

        for gesture_name in os.listdir(split_path):
            gesture_path = os.path.join(split_path, gesture_name)
            if not os.path.isdir(gesture_path):
                continue

            for class_label in ["correct", "wrong"]:
                label_path = os.path.join(gesture_path, class_label)
                if not os.path.exists(label_path):
                    continue

                label_val = 1 if class_label == "correct" else 0
                for img_name in os.listdir(label_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(label_path, img_name)
                        rows.append([img_path, gesture_name, class_label, label_val, split])

    # Save to pairs.csv
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "gesture", "class_name", "label", "split"])
        writer.writerows(rows)

    print(f"Generated {output_csv} with {len(rows)} entries.")
    return output_csv

if __name__ == "__main__":
    generate_pairs_csv()
