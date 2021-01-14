import numpy as np
from pathlib import Path

import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET

DIGIT_CLASSES = [str(i) for i in range(0, 10)] + ['-', 'UNK1', 'UNK2']
DIGIT_DIRNAME = Path('datasets/voc_digits')


def parse_args():
    global DIGIT_CLASSES
    global DIGIT_DIRNAME
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 20],
                        help="Range of seeds")
    parser.add_argument('--ext', type=str, default='png')
    parser.add_argument('--train-size', type=float, default=0.75)
    args = parser.parse_args()

    DIGIT_DIRNAME = Path(args.data_dir)
    lines = [line.strip() for line in open(
        DIGIT_DIRNAME.joinpath('class_names.txt'), 'rt').readlines()]
    DIGIT_CLASSES = lines[1:]  # ignore __background__ class
    return args


def generate_train_val_split(args):
    def train_val_split(paths, seed, train_size):
        random.seed(seed)
        paths = paths[:]  # Clone paths
        random.shuffle(paths)
        num_train_sample = np.ceil(train_size * len(paths)).astype(np.int)
        return paths[:num_train_sample], paths[num_train_sample:]

    image_dir = Path(DIGIT_DIRNAME).joinpath('JPEGImages')
    output_dir = Path(DIGIT_DIRNAME).joinpath('ImageSets/Main')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_train_path = output_dir.joinpath('train.txt')
    output_val_path = output_dir.joinpath('val.txt')

    image_paths = sorted(list(image_dir.glob(f'*.{args.ext}')))
    image_names = [path.stem for path in image_paths]
    train_paths, val_paths = train_val_split(image_names, args.seeds[0], args.train_size)

    with open(output_train_path, 'wt') as f:
        for line in train_paths:
            print(line, file=f)

    with open(output_val_path, 'wt') as f:
        for line in val_paths:
            print(line, file=f)


def generate_seeds(args):
    data_per_cat = {c: [] for c in DIGIT_CLASSES}
    for anno_file in DIGIT_DIRNAME.joinpath('Annotations').glob('*.xml'):
        tree = ET.parse(anno_file)
        clses = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            clses.append(cls)
        for cls in set(clses):
            data_per_cat[cls].append(anno_file)

    result = {cls: {} for cls in data_per_cat.keys()}
    shots = [1, 2, 3, 5, 10]
    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        for c in data_per_cat.keys():
            c_data = []
            for j, shot in enumerate(shots):
                diff_shot = shots[j] - shots[j-1] if j != 0 else 1
                if len(data_per_cat[c]) <= shot:
                    result[c][shot] = copy.deepcopy(data_per_cat[c])
                else:
                    shots_c = random.sample(data_per_cat[c], diff_shot)
                    num_objs = 0
                    for s in shots_c:
                        if s not in c_data:
                            tree = ET.parse(s)
                            file = tree.find("filename").text
                            name = DIGIT_DIRNAME.joinpath('JPEGImages', file)
                            c_data.append(name)
                            for obj in tree.findall("object"):
                                if obj.find("name").text == c:
                                    num_objs += 1
                            if num_objs >= diff_shot:
                                break
                    result[c][shot] = copy.deepcopy(c_data)
        save_path = Path(args.output_dir).joinpath(f'seed{i}')
        save_path.mkdir(parents=True, exist_ok=True)
        for c in result.keys():
            for shot in result[c].keys():
                filename = 'box_{}shot_{}_train.txt'.format(shot, c)
                with open(save_path.joinpath(filename), 'w') as fp:
                    fp.write('\n'.join(list(map(str, result[c][shot])))+'\n')


if __name__ == '__main__':
    args = parse_args()
    generate_train_val_split(args)
    generate_seeds(args)
