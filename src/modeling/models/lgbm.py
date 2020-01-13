
from modeling.models.my_classifier import MyClassifier
import re
import numpy as np
from sklearn.svm import SVC
import lightgbm as lgb
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
from modeling.features import get_all_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from integration.tools import *
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class Lgbm_Classification(MyClassifier):
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
        '''
        params = {
            # "objective": "multiclass",
            "objective": "regression",
            # "metric": "multi_error",
            "metric": "l1",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4,
            # "num_class": max_score + 1
        }
        '''


        params = {
            "objective": "multiclass",
            # "objective": "regression",
            "metric": "multi_error",
            # "metric": "l1",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4,
            "num_class": max_score + 1
        }


        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_dev, label=y_dev)

        hyperparams = self.hyperopt(x_train, y_train, params)

        # x_, x_val, y_, y_val = train_test_split(train_data, valid_data, test_size=0.1, random_state=None)

        self._model = lgb.train({**params, **hyperparams}, train_data, 500, valid_data, early_stopping_rounds=30, verbose_eval=100)

    @ignore_warnings(category=ConvergenceWarning)
    def predict(self, x_test):
        pred_y = self._model.predict(x_test)
        # log(f'Head 10 of prediction_lgbm:\n{pred_y[:10]}')

        pred_y = np.argmax(pred_y, axis=1)
        # pred_y = np.round(pred_y)

        log(f'Head 10 of prediction_lgbm:\n{pred_y[:10]}')
        return pred_y

    @ignore_warnings(category=ConvergenceWarning)
    def hyperopt(self, X, y, params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=None)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
            "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6, 7, 8]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        @ignore_warnings(category=ConvergenceWarning)
        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 300,
                              valid_data, early_stopping_rounds=30, verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]

            return {'loss': score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=100, verbose=1,
                    rstate=np.random.RandomState(1))
        hyperparams = space_eval(space, best)

        log(f"error = {trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams


class Lgbm_Regressor(MyClassifier):
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
            # "objective": "multiclass",
            "objective": "regression",
            # "metric": "multi_error",
            "metric": "l1",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4,
            # "num_class": max_score + 1
        }

        '''
        params = {
            "objective": "multiclass",
            # "objective": "regression",
            "metric": "multi_error",
            # "metric": "l1",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4,
            "num_class": max_score + 1
        }
        '''

        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_dev, label=y_dev)

        hyperparams = self.hyperopt(x_train, y_train, params)

        # x_, x_val, y_, y_val = train_test_split(train_data, valid_data, test_size=0.1, random_state=None)

        self._model = lgb.train({**params, **hyperparams}, train_data, 500, valid_data, early_stopping_rounds=30, verbose_eval=100)

    @ignore_warnings(category=ConvergenceWarning)
    def predict(self, x_test):
        pred_y = self._model.predict(x_test)
        # log(f'Head 10 of prediction_lgbm:\n{pred_y[:10]}')

        # pred_y = np.argmax(pred_y, axis=1)
        pred_y = np.round(pred_y)

        log(f'Head 10 of prediction_lgbm:\n{pred_y[:10]}')
        return pred_y

    @ignore_warnings(category=ConvergenceWarning)
    def hyperopt(self, X, y, params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=None)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
            "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6, 7, 8]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        @ignore_warnings(category=ConvergenceWarning)
        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 300,
                              valid_data, early_stopping_rounds=30, verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]

            return {'loss': score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=100, verbose=1,
                    rstate=np.random.RandomState(1))
        hyperparams = space_eval(space, best)

        log(f"error = {trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams
