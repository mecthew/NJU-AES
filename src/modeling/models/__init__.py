#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/11/16 16:40
# @Author:  Mecthew

from .my_classifier import MyClassifier
from .svm import Svm
from .svr import Svr
from .lr import Lr
from .random_forest import Rf
from .lgbm import Lgbm_Classification, Lgbm_Regressor
from .xgb import Xgboost_Classifier, Xgboost_Regressor
from .gbr import Gbr
from .mlp import Mlp
from .lstm import Lstm, LstmMulInputs
from .cnn import Cnn, CnnMulInputs
from .crnn import Crnn
from .attention import Attention

SVM_MODEL = 'svm'
SVR_MODEL = 'svr'
LR_MODEL = 'lr'
LGBM_MODEL = 'lgbm'
RF_MODEL = 'rf'
XGB_MODEL = 'xgb'
GBR_MODEL = 'gbr'
MLP_MODEL = 'mlp'
LSTM_MODEL = 'lstm'
LSTM_MULTIPLE = 'lstm_mul'
CNN_MODEL = 'cnn'
CNN_MULTIPLE = 'cnn_mul'
CRNN_MODEL = 'crnn'
