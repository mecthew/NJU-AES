
from modeling.models.my_classifier import MyClassifier
import numpy as np
# from sklearn.linear_model import logistic
from sklearn.linear_model import LogisticRegression
from modeling.features import get_all_features
from sklearn.preprocessing import StandardScaler
from integration.tools import *


class Lr(MyClassifier):
    def __init__(self):
        self.max_length = None
        self._model = None
        self._is_init = False

    def init_model(self, kernel, max_iter=200, **kwargs):
        self._model = LogisticRegression(C=1.0, max_iter=max_iter, solver='liblinear', multi_class='auto')
        self._is_init = True

    @property
    def is_init(self):
        return self._is_init

    def fit(self, x_train, y_train, x_dev, y_dev, train_loop_num, **kwargs):
        self._model.fit(x_train, y_train)

    def predict(self, x_test):
        pred_y = self._model.predict(x_test)
        log(f'Head 10 of prediction_lr:\n{pred_y[:10]}')
        return pred_y

