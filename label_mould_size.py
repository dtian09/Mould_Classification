'''
mould area coverage classification
'''
import os

def get_size_class(width, height):
    size = width + height #max size = 1 + 1 = 2
    if size < 2/3:
        return '1' #1: mould covers <1/3 of whole area
    elif size <= 4/3: 
        return '2' #2: mould covers between 1/3 and 2/3 of whole area
    else:
        return '3' #3: mould covers >2/3 of whole area

def process_labels(label_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as out:
        for fname in os.listdir(label_dir):
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(label_dir, fname)
            with open(fpath, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                out.write(f'{fname}\t0\n') #0: normal (no mould)
                continue
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    out.write(f'{fname}\tinvalid_label\n')
                    continue
                width = float(parts[3])
                height = float(parts[4])
                size_class = get_size_class(width, height)
                out.write(f'{fname}\t{size_class}\n')

def main():
    base = 'Mould detection single label.v12-phase-1-yolov11.yolov7pytorch'
    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(base, split, 'labels')
        output_file = f'{split}_mould_size_labels.txt'
        process_labels(label_dir, output_file)
    print('Mould area coverage classification complete.')

if __name__ == '__main__':
    main() 