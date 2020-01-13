#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/11/16 15:23
# @Author:  Mecthew
"""integration program for AES"""

import os
import sys
from os.path import join
from sys import path
import argparse
import time
import numpy as np

from integration.common import get_logger, Timer
from integration.dataset import AESDataset
from integration.metrics import kappa
from integration.tools import log_prompt, log
from modeling.model import Model
from modeling.constant import IS_CROSS_DATASET, PROMPT_NUM

sys.path.append("../../..")

# Verbosity level of logging:
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


def _here(*args):
    """Helper function for getting the current directory of this script."""
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(here, *args))


def _parse_args():
    root_dir = os.path.join(_here(os.pardir), os.pardir)
    default_dataset_dir = join(root_dir, "essay_data/DEMO")
    default_output_dir = join(root_dir, "result_output")
    default_integration_dir = join(root_dir, "src/integration")
    default_modeling_dir = join(root_dir, "src/modeling")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default=default_dataset_dir,
                        help="Directory storing the dataset (containing "
                             "e.g. adult.data/)")
    parser.add_argument('--output_dir', type=str,
                        default=default_output_dir,
                        help="Directory storing the predictions. It will "
                             "contain e.g. [start.txt, adult.predict_0, "
                             "adult.predict_1, ..., end.txt] when integration "
                             "terminates.")
    parser.add_argument('--integration_dir', type=str,
                        default=default_integration_dir,
                        help="Directory storing the integration program "
                             "`integration.py` and other necessary packages.")
    parser.add_argument('--modeling_dir', type=str,
                        default=default_modeling_dir,
                        help="Directory storing the submission code "
                             "`model.py` and other necessary packages.")

    args = parser.parse_args()
    LOGGER.debug(f'Parsed args are: {args}')
    LOGGER.debug("-" * 50)
    return args


def _init_python_path(args):
    path.append(args.integration_dir)
    path.append(args.modeling_dir)
    os.makedirs(args.output_dir, exist_ok=True)


def _check_umodel_methed(umodel):
    # Check if the model has methods `train`, `predict`.
    for attr in ['train', 'predict']:
        if not hasattr(umodel, attr):
            raise Exception("Your model object doesn't have the method "
                            f"`{attr}`. Please implement it in model.py.")


def _train(umodel, train_set, dev_set):
    # Train the model
    timer = Timer()
    # x_train, y_train = dataset
    umodel.train(train_set, dev_set)

    duration = timer.get_duration()
    LOGGER.info(f"Finished training the model. time spent {duration} sec")

    # return _dev(umodel, dev_set)


def _dev(umodel, dataset):
    # Validate on model
    timer = Timer()
    x_dev, y_dev = dataset
    y_pred = umodel.predict(x_dev, is_dev=True)
    score = kappa(y_true=y_dev, y_pred=y_pred)

    duration = timer.get_duration()
    LOGGER.info(f"Finished validating the model. time spent {duration} sec. Validate score: {score}")
    return score


def _predict(umodel, dataset):
    # Make predictions using the trained model
    timer = Timer()
    essay, essay_id, essay_set = dataset
    y_pred, y_pred_dev = umodel.predict(essay, is_dev=False)

    # log(f"y_pred_test_dev:{y_pred_test_dev}")

    y_pred = np.round(y_pred)

    duration = timer.get_duration()
    LOGGER.info(f"Finished predicting test essay. time spent {duration} sec")
    assert len(essay_id) == len(essay_set) and len(essay_set) == len(y_pred)
    return (essay_id, essay_set, y_pred), y_pred_dev


def do_ingestion():
    """main entry"""
    LOGGER.info('===== Start integration program.')
    # Parse directories from input arguments
    LOGGER.info('===== Initialize args.')
    args = _parse_args()
    _init_python_path(args)

    dataset = AESDataset(args.dataset_dir, prompt_num=PROMPT_NUM, is_cross_dataset=IS_CROSS_DATASET)
    x_train_list, y_train_list = dataset.get_train()
    x_dev_list, y_dev_list = dataset.get_dev()
    essay_list, essay_id_list, essay_set_list = dataset.get_test()

    score_list = []
    prediction_list = []
    for i in range(PROMPT_NUM):
        log_prompt(entry="Begin handling ", prompt=i+1)
        x_train, y_train = x_train_list[i], y_train_list[i]
        x_dev, y_dev = x_dev_list[i], y_dev_list[i]
        essay, essay_id, essay_set = essay_list[i], essay_id_list[i], essay_set_list[i]
        umodel = Model(prompt=i+1, max_iter=1)
        # LOGGER.info("===== Check model methods =====")
        # _check_umodel_methed(umodel)

        dev_score, pred_result = None, None
        while not umodel.done_training:
            LOGGER.info(f"===== Begin training model =====")
            _train(umodel, (x_train, y_train), (x_dev, y_dev))

            LOGGER.info("===== Begin predicting on test set =====")
            pred_result, pred_result_dev = _predict(umodel, (essay, essay_id, essay_set))

        pred_result_dev = np.round(pred_result_dev)
        dev_score = kappa(y_true=y_dev, y_pred=pred_result_dev)

        log(f"--------------Prompt{i+1} is done, and the dev_score is {dev_score}-------------")

        score_list.append(dev_score)
        prediction_list.append(pred_result)

    # save result
    score_file = os.path.join(args.output_dir, "score-" + time.strftime("%Y-%m-%d@%H-%M-%S") + '.txt')
    prediction_file = os.path.join(args.output_dir, "prediction-" + time.strftime("%Y-%m-%d@%H-%M-%S") + '.txt')
    LOGGER.info("===== Begin Saving prediction =====")
    # with open(score_file, 'w', encoding='utf8') as fout:
    #     score_list = [str(score) for score in score_list]
    #     fout.write('\n'.join(score_list) + '\n')
    with open(prediction_file, 'w', encoding='utf') as fout:
        for prediction in prediction_list:
            for idx in range(len(prediction[0])):
                fout.write(str(prediction[0][idx]) + '\t' + str(prediction[1][idx])
                           + '\t' + str(prediction[2][idx]) + '\n')
    with open(score_file, 'w', encoding='utf') as fout1:
        tot = 0.0
        for idx in range(len(score_list)):
            tot += score_list[idx]
            fout1.write(str(idx + 1) + '\t' + str(score_list[idx]) + '\n')
        avg = tot * 1.0 / PROMPT_NUM
        fout1.write("avg_score: " + str(avg) + '\n')

    LOGGER.info("[Ingestion terminated]")


if __name__ == '__main__':
    do_ingestion()
