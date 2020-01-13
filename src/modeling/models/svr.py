from modeling.models.my_classifier import MyClassifier
import numpy as np
from sklearn.svm import SVR
from modeling.features import get_all_features
from sklearn.preprocessing import StandardScaler
from integration.tools import *


class Svr(MyClassifier):
    def __init__(self):
        self._model = None
        self._is_init = False

    def init_model(self, kernel, max_iter, C=1.0, gamma='auto', **kwargs):
        self._model = SVR(C=C, gamma=gamma, kernel=kernel, max_iter=max_iter)
        self._is_init = True

    @property
    def is_init(self):
        return self._is_init

    def fit(self, x_train, y_train, x_dev, y_dev, train_loop_num, **kwargs):

        self._model.fit(x_train, y_train)

    def predict(self, x_test):
        pred_y = self._model.predict(x_test)
        log(f'Head 10 of prediction_svr_0:\n{pred_y[:10]}')

        # pred_y = np.round(pred_y)

        # log(f'Head 10 of prediction_svr_1:\n{pred_y[:10]}')

        return pred_y


