import os
import yaml
import re

archive_dir = "archive (2)"
yaml_path = os.path.join(archive_dir, "dataset.yaml")

with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

classes = data.get("names", [])
print(f"Total defined classes in YAML: {len(classes)}")

train_dir = os.path.join(archive_dir, "images", "train")
val_dir = os.path.join(archive_dir, "images", "val")

print(f"Train dir exists: {os.path.exists(train_dir)}")
print(f"Val dir exists: {os.path.exists(val_dir)}")

train_files = os.listdir(train_dir) if os.path.exists(train_dir) else []
print(f"Total train images: {len(train_files)}")
print(f"Sample train images: {train_files[:5]}")

# Map filenames to classes
# Let's see how class names match filenames
class_counts = {c: 0 for c in classes}
unmatched = []

# Sort classes by descending length so longer prefixes match first (e.g., Katakavardhana before Kataka)
sorted_classes = sorted(classes, key=lambda x: len(x), reverse=True)

for fname in train_files:
    fname_clean = fname.split('(')[0].split('_')[0].strip()
    matched = False
    for c in sorted_classes:
        if fname.lower().startswith(c.lower()):
            class_counts[c] += 1
            matched = True
            break
    if not matched:
        unmatched.append(fname)

print("\n--- Class Counts Summary ---")
for c, cnt in class_counts.items():
    if cnt > 0:
        print(f"  {c}: {cnt}")

zero_classes = [c for c, cnt in class_counts.items() if cnt == 0]
print(f"\nClasses with 0 images: {len(zero_classes)} -> {zero_classes}")
print(f"Total matched: {sum(class_counts.values())}")
print(f"Total unmatched: {len(unmatched)}")
if unmatched:
    print(f"Sample unmatched filenames: {unmatched[:10]}")
