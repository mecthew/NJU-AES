import numpy as np
from integration.tools import log
from integration.metrics import kappa
from modeling.models import *
from modeling.features import get_all_features, get_features, get_features_test


class ModelManager:
    def __init__(self,
                 train_set,
                 dev_set,
                 keep_num=5,
                 patience=5,
                 *args,
                 **kwargs):

        self._keep_num = keep_num
        self._patience = patience
        self._train_set = train_set
        self._dev_set = dev_set

        train_x, train_y = train_set

        self._min_score = min(train_y)
        self._max_score = max(train_y)
        self._gap = self._max_score - self._min_score

        # arguments for feature creation
        self._use_tfidf = True
        self._use_fea_select = True

        # record saved by feature selection
        self._X_features = None
        self._scalar_pure = None
        self._scalar_tfidf = None
        self._selector = None
        self._vectorizer = None

        self._model_lib = {
            # TODO: ADD MODELS
            # RF_MODEL: Rf,
            # LGBM_MODEL: Lgbm_Classification,
            # XGB_MODEL: Xgboost_Classifier,
            # XGB_MODEL: Xgboost_Regressor,
            # SVM_MODEL: Svm,
            # GBR_MODEL: Gbr
            SVR_MODEL: Svr,
            # LR_MODEL: Lr,
            # LGBM_MODEL: Lgbm_Regressor,
            # XGB_MODEL: (lambda x: Xgboost_Regressor if self._gap > 10 else Xgboost_Classifier)
        }

        self._input_shape = None
        self._model = None
        self._model_name = None
        self._last_model_name = None
        self._k_best_kappa = [-1.0]*self._keep_num
        self._k_best_predicts = [-1.0]*self._keep_num
        self._k_best_predicts_dev = [-1.0]*self._keep_num

        # create features of train_data in initial module
        # self.preprocess_data(self._train_set, is_test=False)

    def preprocess_data(self, dataset, is_test=False):
        if is_test:
            # dev_x, dev_y = dataset
            dev_x = dataset
            return get_features_test(dev_x,
                                     scalar_pure=self._scalar_pure,
                                     scalar_tfidf=self._scalar_tfidf,
                                     selector=self._selector,
                                     vectorizer=self._vectorizer)
        else:
            train_x, train_y = dataset
            self._X_features, self._scalar_pure, self._scalar_tfidf, \
                self._selector, self._vectorizer = get_features(train_x, train_y, None, 15, self._use_tfidf, self._use_fea_select)

    def _get_or_create_model(self, is_reset_model=False):
        # use new model and not reset model, have to initialize the model
        if self._model_name != self._last_model_name or is_reset_model:
            log(f'Get new model {self._model_name}')
            # TODO: init model parameters
            if self._model_name in [SVM_MODEL, SVR_MODEL]:
                kwargs = {
                    # 'kernel': 'linear',
                    'kernel': 'precomputed',
                    'max_iter': 500
                }
            elif self._model_name in [LR_MODEL]:
                kwargs = {
                    'kernel': 'libinear',
                    'max_iter': 300
                }
            elif self._model_name in [RF_MODEL]:
                kwargs = {
                    'n_estimators': 150,
                    'max_depth': 18,
                    'max_features': 'auto'
                }
            elif self._model_name in [LGBM_MODEL]:
                kwargs = {}
            elif self._model_name in [XGB_MODEL]:
                kwargs = {}
            elif self._model_name in [GBR_MODEL]:
                kwargs = {}
            else:
                raise Exception("No such model!")

            if not self._model.is_init:
                log(f'Init model {self._model_name}')
                self._model.init_model(**kwargs)

    def _pre_select_model(self, train_loop_num, is_reset_model=False):
        self._last_model_name = self._model_name
        model_idx = (train_loop_num-1) % self._model_lib.__len__()
        self._model_name = list(self._model_lib.keys())[model_idx]

        if self._model_name != self._last_model_name or is_reset_model:
            self._model = self._model_lib[self._model_name]()

    def _blending_ensemble(self):
        selected = [idx for idx, _ in enumerate(self._k_best_kappa) if self._k_best_kappa[idx] != -1]
        log(f"Select best {self._keep_num} models" +
            f" which have kappa {[self._k_best_kappa[i] for i in selected]}")
        y_test_ensemble = np.mean([self._k_best_predicts[i] for i in selected], axis=0)
        y_dev_ensemble = np.mean([self._k_best_predicts_dev[i] for i in selected], axis=0)
        # return np.mean([self._k_best_predicts[i] for i in selected], axis=0)
        return y_test_ensemble, y_dev_ensemble

    def fit(self, train_loop_num=1, **kwargs):
        # select model first, inorder to use preprocess data method
        self._pre_select_model(train_loop_num)

        # pre-process data
        train_x, train_y = self._train_set
        # train_x = self._X_features
        # train_x = self._model.preprocess_data(train_x)

        self._input_shape = train_x.shape

        dev_x, dev_y = self._dev_set
        # dev_x = self._model.preprocess_data(dev_x, dev_y)
        # dev_x = self.preprocess_data(dev_x, is_test=True)

        log(f'train_x: {train_x.shape}; train_y: {train_y.shape};'
            f'dev_x: {dev_x.shape}; dev_y: {dev_y.shape};')

        log(f"train_x_features:\n{train_x[:10]}")
        log(f"dev_x_features:\n{dev_x[:10]}")

        # init model really
        self._get_or_create_model(is_reset_model=False)
        self._model.fit(train_x, train_y, dev_x, dev_y, train_loop_num, max_score=self._max_score, **kwargs)

    def predict(self, x_test, is_dev):
        # x_test = self._model.preprocess_data(x_test)
        # x_test = self.preprocess_data(x_test, is_test=True)
        y_pred = self._model.predict(x_test)
        if is_dev:
            return y_pred

        x_dev, y_dev = self._dev_set
        # x_dev = self.preprocess_data(x_dev, is_test=True)
        y_pred_dev = self._model.predict(x_dev)
        kap = kappa(y_true=y_dev, y_pred=y_pred_dev)

        if self._k_best_kappa[-1] < kap:
            self._k_best_kappa[-1] = kap
            self._k_best_predicts[-1] = y_pred
            self._k_best_predicts_dev[-1] = y_pred_dev

        # sort k_best
        for idx, (kap, pred, pred_dev) in enumerate(sorted(
                zip(self._k_best_kappa, self._k_best_predicts, self._k_best_predicts_dev), key=lambda x: x[0], reverse=True)):
            self._k_best_kappa[idx] = kap
            self._k_best_predicts[idx] = pred
            self._k_best_predicts_dev[idx] = pred_dev

        return self._blending_ensemble()

