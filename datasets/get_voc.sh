#!/bin/bash

export datasets='/home/lvloi/few-shot-object-detection/datasets/VOC'

# VOC2007 DATASET
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar # VOC2007 train+val set
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar # VOC2007 test set
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar # VOC2007 devkit
# VOC2012 DATASET
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # VOC2012 train+val set
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar  # VOC2012 devkit

# VOC2007 data:
mkdir $datasets/VOC2007
mkdir $datasets/VOC2007/VOCdevkit
tar xvf VOCtrainval_06-Nov-2007.tar  -C $datasets/VOC2007/VOCdevkit
tar xvf VOCtest_06-Nov-2007.tar -C $datasets/VOC2007/VOCdevkit
tar xvf VOCdevkit_08-Jun-2007.tar -C $datasets/VOC2007/VOCdevkit
# VOC2012 data:
mkdir $datasets/VOC2012
mkdir $datasets/VOC2012/VOCdevkit
tar xvf VOCtrainval_11-May-2012.tar -C $datasets/VOC2012/VOCdevkit
tar xvf VOCdevkit_18-May-2011.tar -C $datasets/VOC2012/VOCdevkit