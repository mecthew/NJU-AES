#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/12/15 20:04
# @Author:  Mecthew
from abc import ABC

from modeling.models import MyClassifier
from keras import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
import keras
from keras.utils import to_categorical
import numpy as np


class Mlp(MyClassifier, ABC):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout=0.0,
                 use_regression=True):
        self._use_regression = use_regression
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropout = dropout
        self._model = None
        self._is_init = False

    @property
    def is_init(self):
        return self._is_init

    def _init_model(self):
        model = Sequential()
        # Densely Connected Neural Network (Multi-Layer Perceptron)
        model.add(Dense(128, activation='softplus', kernel_initializer='he_normal', input_dim=self._input_dim))
        model.add(Dense(64, activation='softplus'))
        # model.add(Dense(32, activation='softplus'))
        model.add(Dropout(self._dropout))
        # model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
        # model.add(Dropout(dropout))
        if self._use_regression:
            model.add(Dense(self._output_dim))
        else:
            model.add(Dense(self._output_dim, activation='softmax'))
        model.summary()

        # Compile the model
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        if self._use_regression:
            model.compile(optimizer=adam, loss='mse', metrics=['mse', 'mae'])
        else:
            model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mse'])

        self._model = model
        self._is_init = True

    def fit(self, x_train, y_train, x_dev, y_dev):
        if not self._use_regression:
            y_train = to_categorical(y_train)
            y_dev = to_categorical(y_dev)
        callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5)]
        self._model.fit(
            x_train, y_train,
            epochs=1000,
            callbacks=callbacks,
            validation_data=(x_dev, y_dev),
            verbose=1,  # Logs once per epoch.
            batch_size=32,
            shuffle=True)

    def predict(self, x_test):
        predicts = self._model.predict(x_test, batch_size=32)
        if self._use_regression:
            return predicts
        else:
            return np.argmax(predicts, axis=-1)