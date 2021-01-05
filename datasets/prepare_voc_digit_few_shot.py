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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 20],
                        help="Range of seeds")
    args = parser.parse_args()
    return args


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
        save_path = Path(f'datasets/digit_vocsplit/seed{i}')
        save_path.mkdir(parents=True, exist_ok=True)
        for c in result.keys():
            for shot in result[c].keys():
                filename = 'box_{}shot_{}_train.txt'.format(shot, c)
                with open(save_path.joinpath(filename), 'w') as fp:
                    fp.write('\n'.join(list(map(str, result[c][shot])))+'\n')


if __name__ == '__main__':
    args = parse_args()
    generate_seeds(args)
