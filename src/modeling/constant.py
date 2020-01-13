#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/11/16 16:43
# @Author:  Mecthew
import os

NUM_THREAD = os.cpu_count()-1
ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)

# Default setting
PROMPT_NUM = 8
IS_CROSS_DATASET = False
SCALE_SCORE = False

# Set according to the paper
NGRAM_MIN_LENGTH = 1
NGRAM_MAX_LENGTH = 15

HISK_DIR = os.path.join(ROOT_DIR, 'essay_data/HISK')
LIBSVM_DIR = os.path.join(HISK_DIR, 'libsvm-input')
SVM_SCALE_DIR = os.path.join(HISK_DIR, 'svm-scale')

TRAIN_LEN = 7787
DEV_LEN = 2593
TEST_LEN = 2598
TEXT_DIM = 300

MAX_SEQ_LEN = 300
DEBUG = True
NUM_EPOCH = 500
