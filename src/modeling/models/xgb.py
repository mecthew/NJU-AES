

from modeling.models.my_classifier import MyClassifier
import re
import numpy as np
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
from modeling.features import get_all_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from integration.metrics import kappa
from integration.tools import *
from xgboost import XGBClassifier, XGBRegressor
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class Xgboost_Classifier(MyClassifier):
    def __init__(self):
        self._model = None
        self._is_init = False
        self._loss = 'multi:softmax'
        self._metrics = "merror"

    def get_xgb_model(self):
        xgb_model = XGBClassifier(
            booster='gbtree',
            max_depth=9,
            objective=self._loss,
            n_estimators=700,
            gamma=0.1,
            min_child_weight=50,
            eta=0.01,
            subsample=0.5,
            colsample_bytree=0.7,
            verbosity=1,
            reg_alpha=0.1,
            nthread=-1
        )
        return xgb_model

    def init_model(self, **kwargs):
        # self._model = self.get_xgb_model()
        self._is_init = True
        log(f"xgb_type:Classification")

    @property
    def is_init(self):
        return self._is_init

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, x_train, y_train, x_dev, y_dev, train_loop_num, max_score, **kwargs):
        params = {
            "booster": 'gbtree',
            "objective": self._loss,
            "verbosity": 1,
            "nthred": -1,
            'reg_alpha': 0.1
        }

        # self._model.fit(x_train, y_train, eval_metric='merror')

        hyperparams = self.hyperopt(x_train, y_train, x_dev, y_dev, params)

        # x_, x_val, y_, y_val = train_test_split(train_data, valid_data, test_size=0.1, random_state=None)

        self._model = XGBClassifier(**params, **hyperparams)
        self._model.fit(X=x_train, y=y_train, eval_metric=self._metrics)

    @ignore_warnings(category=ConvergenceWarning)
    def predict(self, x_test):
        pred_y = self._model.predict(x_test)

        log(f'Head 10 of prediction_xgb:\n{pred_y[:10]}')

        return pred_y

    @ignore_warnings(category=ConvergenceWarning)
    def hyperopt(self, x_train, y_train, x_dev, y_dev, params):
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=None)
        # n_estimators_list = [400, 600, 700, 800, 1000, 1500]
        space = {
            "eta": hp.uniform("eta", 0.1, 1),
            "max_depth": hp.choice("max_depth", [5, 6, 7, 8, 9, 10, 12, 13, 15]),
            "n_estimators": hp.choice("n_estimators", [400, 600, 700, 800, 1000]),
            'subsample': hp.uniform("subsample", 0.5, 1),
            'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
            "min_child_weight": hp.choice("min_child_weight", [1, 2, 3, 4, 5, 6]),
            'gamma': hp.choice('gamma', [0.1, 0.2, 0.3]),
        }

        @ignore_warnings(category=ConvergenceWarning)
        def objective(hyperparams):
            model = XGBClassifier(**params, **hyperparams)

            # model.fit(x_train, y_train, eval_set=[(x_dev, y_dev)], early_stopping_rounds=50, eval_metric='merror', verbose=True)

            model.fit(X=x_train, y=y_train, eval_metric=self._metrics)

            y_pred_dev = model.predict(x_dev)
            score = kappa(y_true=y_pred_dev, y_pred=y_dev)
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()

        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=50, verbose=1,
                    rstate=np.random.RandomState(1))
        hyperparams = space_eval(space, best)

        log(f"kappa = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams


class Xgboost_Regressor(MyClassifier):
    def __init__(self):
        self._model = None
        self._is_init = False
        self._loss = 'reg:linear'
        self._metrics = "rmse"

    def init_model(self, **kwargs):
        log(f"xgb_type:Regressor")
        self._is_init = True

    @property
    def is_init(self):
        return self._is_init

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, x_train, y_train, x_dev, y_dev, train_loop_num, max_score, **kwargs):
        params = {
            "booster": 'gbtree',
            "objective": self._loss,
            "verbosity": 1,
            "nthred": -1,
            'reg_alpha': 0.1
        }

        # self._model.fit(x_train, y_train, eval_metric='merror')

        hyperparams = self.hyperopt(x_train, y_train, x_dev, y_dev, params)

        # x_, x_val, y_, y_val = train_test_split(train_data, valid_data, test_size=0.1, random_state=None)

        self._model = XGBRegressor(**params, **hyperparams)
        self._model.fit(X=x_train, y=y_train, eval_metric=self._metrics)

    @ignore_warnings(category=ConvergenceWarning)
    def predict(self, x_test):
        pred_y = self._model.predict(x_test)
        pred_y = np.round(pred_y)

        log(f'Head 10 of prediction_xgb:\n{pred_y[:10]}')

        return pred_y

    @ignore_warnings(category=ConvergenceWarning)
    def hyperopt(self, x_train, y_train, x_dev, y_dev, params):
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=None)
        # n_estimators_list = [400, 600, 700, 800, 1000, 1500]
        space = {
            "eta": hp.uniform("eta", 0.1, 1),
            "max_depth": hp.choice("max_depth", [5, 6, 7, 8, 9, 10, 12, 13, 15]),
            "n_estimators": hp.choice("n_estimators", [400, 600, 700, 800, 1000]),
            'subsample': hp.uniform("subsample", 0.5, 1),
            'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
            "min_child_weight": hp.choice("min_child_weight", [1, 2, 3, 4, 5, 6]),
            'gamma': hp.choice('gamma', [0.1, 0.2, 0.3]),
        }

        @ignore_warnings(category=ConvergenceWarning)
        def objective(hyperparams):
            model = XGBRegressor(**params, **hyperparams)

            # model.fit(x_train, y_train, eval_set=[(x_dev, y_dev)], early_stopping_rounds=50, eval_metric='merror', verbose=True)

            model.fit(X=x_train, y=y_train, eval_metric=self._metrics)

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
