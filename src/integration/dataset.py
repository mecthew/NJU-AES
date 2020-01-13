"""
  AES datasets.
"""
import os
from datetime import datetime
import json
from glob import glob as ls
import numpy as np
import pandas as pd
from integration.common import get_logger
from integration.metrics import kappa
import csv
from modeling.constant import ROOT_DIR, TRAIN_LEN, DEV_LEN, TEST_LEN


VERBOSITY_LEVEL = 'WARNING'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


class AESDataset:
    """"AESDataset"""
    def __init__(self, dataset_dir, prompt_num=8, is_cross_dataset=False, use_correct=False):
        """
            train_dataset, dev_dataset, test_dataset: list of tuples: (essay_str, score)
        """
        self._dataset_name = dataset_dir
        self._dataset_dir = dataset_dir
        self._train_dataset = None
        self._dev_dataset = None
        self._test_dataset = None
        self._prompt_num = prompt_num
        self._is_cross_dataset = is_cross_dataset
        self._use_correct = use_correct

    def read_dataset(self):
        """read dataset"""
        self._train_dataset = self._read_dataset(
            os.path.join(self._dataset_dir, "train.tsv"))
        self._dev_dataset = self._read_dataset(
            os.path.join(self._dataset_dir, "dev.tsv"))
        self._test_dataset = self._read_dataset(
            os.path.join(self._dataset_dir, "test.tsv"))

    def get_train(self):
        """get train"""
        if self._train_dataset is None:
            self._train_dataset = self._read_dataset(
                os.path.join(self._dataset_dir, "train.tsv"))
        return self._train_dataset

    def get_dev(self):
        """get dev"""
        if self._dev_dataset is None:
            self._dev_dataset = self._read_dataset(
                os.path.join(self._dataset_dir, "dev.tsv"))
        return self._dev_dataset

    def get_test(self):
        """get test"""
        if self._test_dataset is None:
            self._test_dataset = self._read_dataset(
                os.path.join(self._dataset_dir, "test.tsv"))
        return self._test_dataset

    def _read_dataset(self, dataset_path):
        dataset = pd.read_csv(
            dataset_path,
            sep='\t',
            header=0,
            quoting=csv.QUOTE_NONE
        )
        all_essays = []
        corrects_path = ROOT_DIR + '/essay_data/corrects/all_essays.txt'
        if os.path.exists(corrects_path):
            fin = open(corrects_path, 'r', encoding='utf8')
            for line in fin:
                all_essays.append(line.strip())

        if self._is_cross_dataset:
            if dataset_path.endswith('test.tsv'):
                # test.tsv needs return essay_id, essay_set and essay content
                essays, essay_id, essay_set = dataset['essay'].astype(str).values, dataset['essay_id'].astype(int).values,\
                       dataset['essay_set'].astype(int).values,
                new_essays = []
                for each_essay in essays:
                    if each_essay[0] == each_essay[-1] == '\"':
                        each_essay = each_essay[1:-1]
                    new_essays.append(each_essay.strip())
                if not self._use_correct:
                    return new_essays, essay_id, essay_set
                else:
                    return all_essays[-TEST_LEN:], essay_id, essay_set
            else:
                x, y = dataset['essay'].astype(str).values, dataset['domain1_score'].astype(int).values
                new_x = []
                for each_essay in x:
                    if each_essay[0] == each_essay[-1] == '\"':
                        each_essay = each_essay[1:-1]
                    new_x.append(each_essay.strip())

                if not self._use_correct:
                    return np.asarray(new_x), y
                elif dataset_path.endswith('train.tsv'):
                    return all_essays[:TRAIN_LEN], y
                elif dataset_path.endswith('dev.tsv'):
                    return all_essays[TRAIN_LEN:TRAIN_LEN+DEV_LEN], y

        else:
            if dataset_path.endswith('test.tsv'):
                # test.tsv needs return essay_id, essay_set and essay content
                essays, essay_id, essay_set = [], [], []
                for i in range(self._prompt_num):
                    prompt_i = dataset.loc[dataset['essay_set'] == i+1]
                    indexs = prompt_i.index.tolist()
                    essays_i = prompt_i['essay'].astype(str).values
                    new_essays_i = []
                    for each_essay in essays_i:
                        if each_essay[0] == each_essay[-1] == '\"':
                            each_essay = each_essay[1:-1]
                        new_essays_i.append(each_essay.strip())

                    if self._use_correct:
                        new_essays_i = [all_essays[TRAIN_LEN+DEV_LEN+idx] for idx in indexs]
                    essays.append(np.array(new_essays_i))
                    essay_id.append(prompt_i['essay_id'].astype(int).values)
                    essay_set.append(prompt_i['essay_set'].astype(int).values)
                return essays, essay_id, essay_set

            else:
                x, y = [], []
                for i in range(self._prompt_num):
                    prompt_i = dataset.loc[dataset['essay_set'] == i+1]
                    indexs = prompt_i.index.tolist()
                    x_i, y_i = prompt_i['essay'].astype(str).values, prompt_i['domain1_score'].astype(int).values
                    new_x_i = []
                    for each_essay in x_i:
                        if each_essay[0] == each_essay[-1] == '\"':
                            each_essay = each_essay[1:-1]
                        new_x_i.append(each_essay.strip())
                        if self._use_correct and dataset_path.endswith('train.tsv'):
                            new_x_i = [all_essays[idx] for idx in indexs]
                        if self._use_correct and dataset_path.endswith('dev.tsv'):
                            new_x_i = [all_essays[TRAIN_LEN + idx] for idx in indexs]
                    x.append(np.array(new_x_i))
                    y.append(y_i)
                return x, y

    def get_train_num(self):
        """ return the number of train instance """
        assert len(self._train_dataset[0]) == len(self._train_dataset[1])
        return len(self._train_dataset[0])

    def get_dev_num(self):
        """ return the number of dev instance """
        assert len(self._dev_dataset[0]) == len(self._dev_dataset[1])
        return len(self._dev_dataset[0])

    def get_test_num(self):
        """ return the number of test instance """
        return len(self._test_dataset[0])

