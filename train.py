import argparse
import os
import csv
import json
import cv2
import torch
import numpy as np
from services.trainer import train_landmark_mlp
from core.landmarks import normalize_multi_hand_landmarks, augment_landmarks, normalize_landmarks
from adapters.hand_detector import HandDetector

# 47 Verified Mudra Classes from the dataset
MUDRA_CLASSES = [
    'Alapadmam', 'Anjali', 'Aralam', 'Ardhachandran', 'Ardhapathaka',
    'Berunda', 'Bramaram', 'Chakra', 'Chandrakala', 'Chaturam',
    'Garuda', 'Hamsapaksha', 'Hamsasyam', 'Kangulam', 'Kapith',
    'Kapotham', 'Karkatta', 'Kartariswastika', 'Katakamukha', 'Katakavardhana',
    'Katrimukha', 'Khatva', 'Kilaka', 'Kurma', 'Matsya',
    'Mayura', 'Mrigasirsha', 'Mukulam', 'Mushti', 'Nagabandha',
    'Padmakosha', 'Pasha', 'Pathaka', 'Pushpaputa', 'Sakata',
    'Samputa', 'Sarpasirsha', 'Shanka', 'Shivalinga', 'Shukatundam',
    'Sikharam', 'Simhamukham', 'Suchi', 'Swastikam', 'Tamarachudam',
    'Trishulam', 'Varaha'
]

def map_filename_to_class(fname: str, sorted_classes: list):
    """Matches filename to mudra class using sorted prefix matching."""
    for c in sorted_classes:
        if fname.lower().startswith(c.lower()):
            return c
    return None

def extract_landmarks_from_archive(
    archive_dir: str = "archive (2)",
    samples_per_class: int = 40,
    val_samples_per_class: int = 10,
    cache_path: str = "outputs/mudra_landmarks_cache.npz",
    force_extract: bool = False
):
    """
    Extracts multi-hand landmarks (126 dims) from archive dataset images.
    Caches results into a .npz file for lightning-fast retraining.
    """
    if os.path.exists(cache_path) and not force_extract:
        print(f"[CACHE] Loading cached landmarks from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        X_train = torch.tensor(data["X_train"], dtype=torch.float32)
        y_train = torch.tensor(data["y_train"], dtype=torch.long)
        X_val = torch.tensor(data["X_val"], dtype=torch.float32)
        y_val = torch.tensor(data["y_val"], dtype=torch.long)
        classes = list(data["classes"])
        print(f"[CACHE] Loaded: Train={len(X_train)} samples, Val={len(X_val)} samples across {len(classes)} classes.")
        return X_train, y_train, X_val, y_val, classes

    train_dir = os.path.join(archive_dir, "images", "train")
    val_dir = os.path.join(archive_dir, "images", "val")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    sorted_classes = sorted(MUDRA_CLASSES, key=lambda x: len(x), reverse=True)
    class_to_idx = {c: i for i, c in enumerate(MUDRA_CLASSES)}

    # Group image paths by class
    def collect_class_files(dir_path):
        grouped = {c: [] for c in MUDRA_CLASSES}
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                c = map_filename_to_class(fname, sorted_classes)
                if c:
                    grouped[c].append(os.path.join(dir_path, fname))
        return grouped

    print(f"Scanning archive dataset in: {archive_dir}")
    train_files_by_class = collect_class_files(train_dir)
    val_files_by_class = collect_class_files(val_dir) if os.path.exists(val_dir) else {}

    detector = HandDetector(static_image_mode=True, max_num_hands=2)

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    print(f"Extracting landmarks (target up to {samples_per_class} train samples per mudra)...")
    for c_idx, c_name in enumerate(MUDRA_CLASSES):
        # Train images
        imgs = train_files_by_class[c_name]
        selected_train = imgs[:samples_per_class] if samples_per_class > 0 else imgs

        extracted_count = 0
        for img_p in selected_train:
            img = cv2.imread(img_p)
            if img is None:
                continue
            all_coords, _ = detector.find_landmarks(img)
            if all_coords is not None and len(all_coords) > 0:
                feat = normalize_multi_hand_landmarks(all_coords, max_hands=2)
                X_train_list.append(feat)
                y_train_list.append(c_idx)
                extracted_count += 1

                # Data Augmentation: add jittered duplicate
                jittered = augment_landmarks(feat, jitter_std=0.015)
                X_train_list.append(jittered)
                y_train_list.append(c_idx)

        # Val images
        val_imgs = val_files_by_class.get(c_name, [])
        selected_val = val_imgs[:val_samples_per_class] if val_samples_per_class > 0 else val_imgs
        for img_p in selected_val:
            img = cv2.imread(img_p)
            if img is None:
                continue
            all_coords, _ = detector.find_landmarks(img)
            if all_coords is not None and len(all_coords) > 0:
                feat = normalize_multi_hand_landmarks(all_coords, max_hands=2)
                X_val_list.append(feat)
                y_val_list.append(c_idx)

        print(f"  [{c_idx+1:02d}/{len(MUDRA_CLASSES):02d}] {c_name:<16}: {extracted_count} detected (augmented -> {extracted_count*2} train)")

    detector.close()

    if len(X_train_list) == 0:
        raise ValueError("No hands detected in any archive images.")

    X_train_np = np.array(X_train_list, dtype=np.float32)
    y_train_np = np.array(y_train_list, dtype=np.int64)
    X_val_np = np.array(X_val_list, dtype=np.float32) if X_val_list else np.empty((0, 126), dtype=np.float32)
    y_val_np = np.array(y_val_list, dtype=np.int64) if y_val_list else np.empty((0,), dtype=np.int64)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=X_train_np,
        y_train=y_train_np,
        X_val=X_val_np,
        y_val=y_val_np,
        classes=np.array(MUDRA_CLASSES)
    )
    print(f"[CACHE] Saved extracted landmarks to {cache_path}")

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_val = torch.tensor(X_val_np, dtype=torch.float32) if len(X_val_np) > 0 else None
    y_val = torch.tensor(y_val_np, dtype=torch.long) if len(y_val_np) > 0 else None

    return X_train, y_train, X_val, y_val, MUDRA_CLASSES

def main():
    parser = argparse.ArgumentParser(description="Train 47-Class Mudra Landmark Classifier")
    parser.add_argument("--archive-dir", type=str, default="archive (2)", help="Path to archive dataset directory")
    parser.add_argument("--samples-per-class", type=int, default=40, help="Max train images sampled per mudra class")
    parser.add_argument("--val-samples", type=int, default=10, help="Max val images sampled per mudra class")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of landmarks from images")
    parser.add_argument("--model-dir", type=str, default="outputs/models/mudra_mlp", help="Output directory for trained model")
    args = parser.parse_args()

    print("=" * 60)
    print("   ARUNODAYA: 47-CLASS BHARATANATYAM MUDRA TRAINING PIPELINE")
    print("=" * 60)

    # 1. Extract or load cached landmarks
    X_train, y_train, X_val, y_val, classes = extract_landmarks_from_archive(
        archive_dir=args.archive_dir,
        samples_per_class=args.samples_per_class,
        val_samples_per_class=args.val_samples,
        force_extract=args.force_extract
    )

    # 2. Train Mudra Landmark MLP
    model_path, meta_path = train_landmark_mlp(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        classes=classes,
        output_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    # Also sync to default predictor model path
    default_dir = "outputs/models/landmark_mlp"
    if args.model_dir != default_dir:
        os.makedirs(default_dir, exist_ok=True)
        import shutil
        shutil.copy(model_path, os.path.join(default_dir, "model.pt"))
        shutil.copy(meta_path, os.path.join(default_dir, "meta.json"))
        print(f"[OK] Synced production weights to: {default_dir}")

    print("\n" + "=" * 60)
    print("   TRAINING COMPLETE: Mudra model ready for real-time inference!")
    print("=" * 60)

if __name__ == "__main__":
    main()
