'''
Oversample a training set:

1. Read all samples and group by class.
2. Determine the size of class 3 (the largest class).
3. Oversample classes 0, 1, 2 to match class 3’s size (with replacement).
4. Bootstraps of class 3: 
   a. Randomly sample with replacement to create a new set of class 3 samples of the same size.
   b. Find class 3 samples not selected in this bootstrap.
   c. Repeat bootstrapping on the unselected set until fewer than 10 remain unselected.
   d. For each of these remaining unselected samples, create 5 replicates.
5. Combine all samples and shuffle.
6. Write to a new file (e.g., train_mould_size_labels_balanced.txt).
'''
import random
from collections import defaultdict

random.seed(42)  # For reproducibility

label_file = "train_mould_area_labels.txt"
output_file = "train_mould_area_labels_oversampled.txt"

# 1. Read and group by class
class_samples = defaultdict(list)
with open(label_file, "r") as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2 and parts[1].isdigit():
            class_samples[int(parts[1])].append(line.strip())

# 2. Determine class 3 size
n_class3 = len(class_samples[3])

# 3. Oversample classes 0, 1, 2 to class 3 size
balanced_samples = []
for cls in [0, 1, 2]:
    samples = class_samples[cls]
    if samples:
        oversampled = random.choices(samples, k=n_class3)
        balanced_samples.extend(oversampled)

# 4. Bootstrap class 3
class3_samples = class_samples[3][:]
all_selected = set()
remaining = set(range(n_class3))
bootstraps = []
threshold = 10
while len(remaining) >= threshold:
    selected_indices = [random.randint(0, n_class3 - 1) for _ in range(n_class3)]
    bootstraps.extend([class3_samples[i] for i in selected_indices])
    all_selected.update(selected_indices)
    remaining = set(range(n_class3)) - all_selected

# For the remaining (<10) class 3 samples, create 5 replicates each
for idx in remaining:
    balanced_samples.extend([class3_samples[idx]] * 5)

# Add all bootstrapped class 3 samples
balanced_samples.extend(bootstraps)

# 5. Shuffle all samples
random.shuffle(balanced_samples)

# 6. Write to new file
with open(output_file, "w") as f:
    for line in balanced_samples:
        f.write(line + "\n")

print(f"Balanced training set written to {output_file}")