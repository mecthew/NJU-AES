#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/11/23 10:59
# @Author:  Mecthew
from modeling.constant import ROOT_DIR, PROMPT_NUM, IS_CROSS_DATASET, LIBSVM_DIR, SVM_SCALE_DIR, DEBUG
from libsvm.svmutil import *
from libsvm.svm import svm_problem, svm_parameter
from libsvm.commonutil import svm_read_problem
from integration.dataset import AESDataset
from integration.ingestion import _parse_args
from integration.metrics import kappa
from sklearn.preprocessing import MinMaxScaler
import re
import os
import numpy as np
import time


def execute_scale_command(y_scale=False):
    if y_scale:
        svm_scale_dir = os.path.join(ROOT_DIR, 'essay_data/HISK/svm-scale-with_y')
    else:
        svm_scale_dir = os.path.join(ROOT_DIR, 'essay_data/HISK/svm-scale')
    os.makedirs(svm_scale_dir, exist_ok=True)
    libsvm_dir = LIBSVM_DIR

    for prompt in range(PROMPT_NUM):
        libsvm_input_file = os.path.join(libsvm_dir, 'prompt@' + str(prompt+1) + '-libsvm.txt')
        scale_output_file = os.path.join(svm_scale_dir, 'prompt@' + str(prompt+1) + '-scale.txt')
        scale_rule_file = os.path.join(svm_scale_dir, 'prompt@' + str(prompt+1) + '-rule.txt')

        if os.path.exists(scale_output_file):
            continue
        path_to_svm_scale = ROOT_DIR + '/resource/svm-scale.exe'
        scale_command = (
            f'{path_to_svm_scale} -l 0 -u 1 -s '
            f'{scale_rule_file} {libsvm_input_file} >> {scale_output_file}'
        )
        os.system(scale_command)


def convert_to_libsvm_input_format(is_contain_test=True, is_scale_y=False, split_train_dev=False):
    hisk_output_dir = os.path.join(ROOT_DIR, 'essay_data/HISK/output-all')
    libsvm_input_dir = LIBSVM_DIR
    os.makedirs(libsvm_input_dir, exist_ok=True)
    args = _parse_args()
    dataset = AESDataset(args.dataset_dir, prompt_num=PROMPT_NUM, is_cross_dataset=IS_CROSS_DATASET)
    x_train_list, y_train_list = dataset.get_train()
    x_dev_list, y_dev_list = dataset.get_dev()
    x_test_list, x_test_ids, x_test_sets = dataset.get_test()

    for file in os.listdir(hisk_output_dir):
        hisk_file = os.path.join(hisk_output_dir, file)
        if is_contain_test:
            libsvm_file = os.path.join(libsvm_input_dir, file.__str__().split('.')[0] + '-libsvm.txt')
        else:
            libsvm_file = os.path.join(libsvm_input_dir, file.__str__().split('.')[0] + '-libsvm-notest.txt')
        if os.path.exists(libsvm_file) and not split_train_dev:
            continue

        prompt_idx = int(re.findall(r'\d+', file.__str__())[0])
        if is_contain_test:
            y_test = np.zeros(shape=len(x_test_list[prompt_idx-1]))
            y_labels = np.concatenate([y_train_list[prompt_idx - 1], y_dev_list[prompt_idx - 1], y_test], axis=0)
        else:
            y_labels = np.concatenate([y_train_list[prompt_idx - 1], y_dev_list[prompt_idx - 1]], axis=0)
        if is_scale_y:
            mm_scaler = MinMaxScaler(feature_range=(0, 1))
            y_labels = mm_scaler.fit_transform(np.array(y_labels).reshape(-1, 1)).reshape(-1)
        if DEBUG:
            print(f'prompt@{prompt_idx} shape: {y_labels.shape}')
        with open(hisk_file, 'r', encoding='utf8') as fin, open(libsvm_file, 'w', encoding='utf8') as fout:
            x_feas = [line.strip().split() for line in fin][:len(y_labels)]
            for idx, x_fea in enumerate(x_feas):
                fea_idx = 1
                fout.write(str(y_labels[idx]))
                for each_fea in x_fea:
                    fout.write(' ' + str(fea_idx) + ':' + each_fea)
                    fea_idx += 1
                fout.write('\n')

        if split_train_dev:
            libsvm_train_file = os.path.join(libsvm_input_dir, file.__str__().split('.')[0] + '-libsvm-train.txt')
            libsvm_dev_file = os.path.join(libsvm_input_dir, file.__str__().split('.')[0] + '-libsvm-dev.txt')
            with open(libsvm_file, 'r', encoding='utf8') as fin:
                fout1, fout2 = open(libsvm_train_file, 'w', encoding='utf8'), open(libsvm_dev_file, 'w', encoding='utf8')
                train_len = len(x_train_list[prompt_idx-1])
                for idx, line in enumerate(fin):
                    if idx < train_len:
                        fout1.write(line.strip() + '\n')
                    else:
                        fout2.write(line.strip() + '\n')
                fout1.close()
                fout2.close()

    execute_scale_command()


def test_v_svr(prompt_idx, gamma=None):
    args = _parse_args()
    dataset = AESDataset(args.dataset_dir, prompt_num=PROMPT_NUM, is_cross_dataset=False)
    x_train_list, y_train_list = dataset.get_train()
    x_dev_list, y_dev_list = dataset.get_dev()
    x_test_list, _, _ = dataset.get_test()
    train_len, dev_len, test_len = len(x_train_list[prompt_idx-1]), len(x_dev_list[prompt_idx-1]),\
                                   len(x_test_list[prompt_idx-1])

    y, x = svm_read_problem(SVM_SCALE_DIR + '/prompt@' + str(prompt_idx) + '-scale.txt')
    x_train, y_train = x[:train_len], y[:train_len]
    x_dev, y_dev = x[train_len:train_len+dev_len], y[train_len:train_len+dev_len]
    x_test = x[train_len+dev_len:]

    if gamma:
        param = f'-s 4 -t 2 -c 1000 -n 0.1 -g {gamma}'
    else:
        param = f'-s 4 -t 2 -c 1000 -n 0.1'
    svm_model = svm_train(y_train+y_dev, x_train+x_dev, param)
    p_label, p_acc, p_val = svm_predict(np.zeros(shape=len(x_test)), x_test, svm_model)
    p_label = np.round(p_label)

    dev_label, dev_acc, dev_val = svm_predict(y_dev, x_dev, svm_model)
    dev_kappa = kappa(y_true=y_dev, y_pred=dev_label, weights='quadratic')
    print(f'Dev kappa: {dev_kappa}')
    return dev_kappa, p_label


def compute_record_kappa():
    result_out = open(ROOT_DIR+'/result_output/kappa' + time.strftime("%Y-%m-%d@%H-%M-%S") + '.txt', 'w',
                      encoding='utf-8')
    sum1, sum2 = 0.0, 0.0
    # for non-y-scale
    gamma_list = [0.01563, 0.00012, 0.125, 0.125,
                  0.00391, 0.00391, 0.01563, 0.03125]
    # for y-scale
    # gamma_list = [0.25, 0.5, 0.25, 0.25,
    #               0.125, 0.25, 0.125, 0.00391]
    all_predicts = []
    for i in range(PROMPT_NUM):
        k1, predicts = test_v_svr(i+1, gamma=gamma_list[i])
        result_out.write(str(k1) + '\n')
        sum1 += k1
        all_predicts.append(list(predicts))

    save_libsvm_predicts(all_predicts=all_predicts)
    result_out.write(f'average: {sum1/PROMPT_NUM}\n')
    result_out.close()


def save_libsvm_predicts(all_predicts):
    fout = open(ROOT_DIR + '/result_output/predicts-libsvm' + time.strftime("%Y-%m-%d@%H-%M-%S") + '.tsv',
                'w', encoding='utf8')
    args = _parse_args()
    dataset = AESDataset(args.dataset_dir, prompt_num=PROMPT_NUM, is_cross_dataset=IS_CROSS_DATASET)
    essay_list, essay_id_list, essay_set_list = dataset.get_test()
    for i in range(PROMPT_NUM):
        essay_ids = essay_id_list[i]
        essay_sets = essay_set_list[i]
        for idx, prediction in enumerate(all_predicts[i]):
            fout.write(f'{essay_ids[idx]}\t{essay_sets[idx]}\t{prediction}\n')
    fout.close()


def grid_search():
    for i in range(1, 9):
        train_file = os.path.join('libsvm', 'prompt@' + str(i) + '-libsvm-train.txt')
        dev_file = os.path.join('libsvm', 'prompt@' + str(i) + '-libsvm-dev.txt')

        scale_file = r'C:\Users\90584\Desktop\Github\AES\essay_data\HISK\svm-scale-with_y\prompt@' + str(i) + '-scale.txt'

        if i != 2:
            continue
        command = f'python easy.py {train_file} {dev_file}'
        exec_com = (
            f'python gridregression.py -log2c 9,10,1 -log2p "null" -s 4 -t 2 -n 0.1 {scale_file}'
        )
        os.system(exec_com)


if __name__ == '__main__':
    convert_to_libsvm_input_format(is_contain_test=True, is_scale_y=False, split_train_dev=False)
    # execute_scale_command()
    os.makedirs(ROOT_DIR + '/result_output', exist_ok=True)
    compute_record_kappa()

