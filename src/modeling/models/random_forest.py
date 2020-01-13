
from modeling.models.my_classifier import MyClassifier
import re
import numpy as np
from sklearn.svm import SVC
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
from sklearn.ensemble import RandomForestClassifier
from modeling.features import get_all_features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from integration.tools import *
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class Rf(MyClassifier):
    def __init__(self):
        self._model = None
        self._is_init = False

    def init_model(self, **kwargs):
        # self._model = RandomForestClassifier(n_estimators=150, max_depth=30)
        self._is_init = True

    @property
    def is_init(self):
        return self._is_init

    def preprocess_data(self, x):

        '''
        REPLACE_BY_SPACE_RE = re.compile(r'["/(){}[\]|@,;.?!]')
        x_tokens = [REPLACE_BY_SPACE_RE.sub(' ', essay).split(' ') for essay in x]
        x_len = np.array(list(map(len, x_tokens)))
        x_len = x_len[:, np.newaxis]
        return x_len
        '''

        features = get_all_features(x)

        x_fea = []
        for i in range(len(x)):
            temp = []
            for key, values in features.items():
                temp.append(values[i])
            temp = np.asarray(temp)
            x_fea.append(temp)

        x_fea = np.asarray(x_fea)
        # log(f"x_fea:\n{x_fea}")

        scaler = StandardScaler()
        X = scaler.fit_transform(x_fea[:, :])

        return X

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, x_train, y_train, x_dev, y_dev, train_loop_num, **kwargs):
        hyperparams_rf = self.hyperopt_rf(x_train, y_train)
        self._model = RandomForestClassifier(**hyperparams_rf, n_jobs=4)
        self._model.fit(x_train, y_train)

    @ignore_warnings(category=ConvergenceWarning)
    def predict(self, x_test):
        pred_y = self._model.predict(x_test)
        log(f'Head 10 of prediction_rf:\n{pred_y[:10]}')
        return pred_y

    @ignore_warnings(category=ConvergenceWarning)
    def hyperopt_rf(self, X, y):
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

        space = {
            'max_depth': hp.choice('max_depth', range(8, 20)),
            # 'max_features': hp.choice('max_features', range(1, 5)),
            'max_features': 'auto',
            'n_estimators': hp.choice('n_estimators', range(100, 150)),
            # 'criterion': hp.choice('criterion', ["gini", "entropy"]),
            # 'scale': hp.choice('scale', [0, 1]),
            # 'normalize': hp.choice('normalize', [0, 1])
        }

        @ignore_warnings(category=ConvergenceWarning)
        def objective(hyperparams):
            model = RandomForestClassifier(**hyperparams, n_jobs=4)

            score = cross_val_score(model, X, y, cv=5, n_jobs=4, scoring='accuracy').mean()

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=70, verbose=1,
                    rstate=np.random.RandomState(1))
        hyperparams = space_eval(space, best)

        log(f"accuracy = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams