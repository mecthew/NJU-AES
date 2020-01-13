#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/11/16 23:39
# @Author:  Mecthew
from modeling.models.my_classifier import MyClassifier
import numpy as np
from sklearn.svm import SVC
from modeling.features import get_all_features, get_features
from sklearn.preprocessing import StandardScaler
from integration.tools import *


class Svm(MyClassifier):
    def __init__(self):
        self._model = None
        self._is_init = False

    def init_model(self, kernel, max_iter, **kwargs):
        self._model = SVC(C=1.0, kernel=kernel, max_iter=max_iter)
        self._is_init = True

    @property
    def is_init(self):
        return self._is_init

    def fit(self, x_train, y_train, x_dev, y_dev, train_loop_num, max_score, **kwargs):
        log(f"svm_x_train_shape:{x_train.shape}")
        log(f"svm_y_train_shape:{y_train.shape}")

        self._model.fit(x_train, y_train)

    def predict(self, x_test):
        pred_y = self._model.predict(x_test)
        log(f'Head 10 of prediction_svm:\n{pred_y[:10]}')
        return pred_y
