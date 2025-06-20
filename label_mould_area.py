'''
mould area coverage classification by total normalized area.
Calculate the normalized total mould area for each image (sum of width × height for all regions).
Assign a category based on the total area using these thresholds:
0: normal (no mould)
1: small (0 < area ≤ 0.05)
2: medium (0.05 < area ≤ 0.15)
3: large (0.15 < area ≤ 0.3)
4: extra large (area > 0.3)
'''
import os

def get_area_category(area):
    if area == 0:
        return '0'  # 0: normal (no mould)
    #elif area <= 0.05:
    #    return '1'  # 1: small
    elif area <= 0.15:
        return '1'  # 1: small or medium
    elif area <= 0.3:
        return '2'  # 2: large
    else:
        return '3'  # 3: extra large

def process_labels(label_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as out:
        for fname in os.listdir(label_dir):
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(label_dir, fname)
            total_area = 0.0
            with open(fpath, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                out.write(f'{fname}\t0\n')  # 0: normal (no mould)
                continue
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue  # skip invalid lines
                width = float(parts[3])
                height = float(parts[4])
                total_area += width * height
            area_category = get_area_category(total_area)
            out.write(f'{fname}\t{area_category}\n')

def main():
    base = 'Mould detection single label.v12-phase-1-yolov11.yolov7pytorch'
    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(base, split, 'labels')
        output_file = f'{split}_mould_area_labels.txt'
        process_labels(label_dir, output_file)
    print('Mould area coverage classification by total area complete.')

if __name__ == '__main__':
    main() 