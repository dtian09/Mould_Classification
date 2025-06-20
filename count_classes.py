from collections import Counter

label_file = "train_mould_area_labels.txt"
#label_file = "test_mould_area_labels.txt"
#label_file = "valid_mould_area_labels.txt"
#label_file = "train_mould_area_labels_oversampled.txt"
class_counts = Counter()
with open(label_file, "r") as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2 and parts[1].isdigit():
            class_counts[int(parts[1])] += 1

for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count} samples")