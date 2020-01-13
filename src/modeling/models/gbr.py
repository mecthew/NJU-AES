
from modeling.models.my_classifier import MyClassifier
import re
import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as gbr
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
from modeling.features import get_all_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from integration.tools import *
from integration.metrics import kappa
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class Gbr(MyClassifier):
    def __init__(self):
        self._model = None
        self._is_init = False

    def init_model(self, **kwargs):
        self._is_init = True

    @property
    def is_init(self):
        return self._is_init

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, x_train, y_train, x_dev, y_dev, train_loop_num, max_score, **kwargs):
        params = {
            "loss": "ls"
        }

        hyperparams = self.hyperopt(x_train, y_train, x_dev, y_dev, params)

        # x_, x_val, y_, y_val = train_test_split(train_data, valid_data, test_size=0.1, random_state=None)

        self._model = gbr(**params, **hyperparams)

        # self._model = gbr(n_estimators=150, learning_rate=0.1, max_depth=10, subsample=0.6, loss='l1')

        self._model.fit(x_train, y_train)

    @ignore_warnings(category=ConvergenceWarning)
    def predict(self, x_test):
        pred_y = self._model.predict(x_test)
        log(f'Head 10 of prediction_lgbm:\n{pred_y[:10]}')

        # pred_y = np.argmax(pred_y, axis=1)
        pred_y = np.round(pred_y)

        log(f'Head 10 of prediction_lgbm:\n{pred_y[:10]}')
        return pred_y

    @ignore_warnings(category=ConvergenceWarning)
    def hyperopt(self, x_train, y_train, x_dev, y_dev, params):

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
            "max_depth": hp.choice("max_depth", [5, 6, 7, 8, 9, 10, 11, 12, 13]),
            "n_estimators": hp.choice('n_estimators', range(100, 150)),
            "subsample": hp.uniform("subsample", 0.5, 0.8),
        }

        @ignore_warnings(category=ConvergenceWarning)
        def objective(hyperparams):
            model = gbr(**params, **hyperparams)

            model.fit(x_train, y_train)

            y_pred_dev = model.predict(x_dev)
            score = kappa(y_true=y_pred_dev, y_pred=y_dev)

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=2, verbose=1,
                    rstate=np.random.RandomState(1))
        hyperparams = space_eval(space, best)

        log(f"kappa = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams

