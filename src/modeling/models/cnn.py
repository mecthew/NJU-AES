#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/12/16 20:22
# @Author:  Mecthew

from modeling.models import MyClassifier
import numpy as np
import keras
from keras import optimizers
from keras.layers import (Input, Dense, Dropout, Convolution2D, LSTM, Bidirectional, Flatten, CuDNNLSTM,
                          MaxPooling2D, ELU, Reshape, CuDNNGRU, Conv1D, MaxPooling1D, AveragePooling1D, Concatenate)
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.models import Model
from modeling.constant import MAX_SEQ_LEN, TEXT_DIM, NUM_EPOCH


class Cnn(MyClassifier):
    def __init__(self):
        self._model = None
        self._is_init = False
        self._model = None

    def preprocess_data(self, x):
        return x

    @property
    def is_init(self):
        return self._is_init

    def init_model(self,
                   input_shape,
                   output_dim,
                   dropout=0.0,
                   **kwargs):

        inputs = Input(shape=input_shape)
        kernel_sizes = [2, 3, 4]
        convs = []
        for i in range(len(kernel_sizes)):
            conv_l = Conv1D(filters=100,
                            kernel_size=(kernel_sizes[i]),
                            kernel_initializer='normal',
                            bias_initializer='random_uniform',
                            activation='relu',
                            padding='same')(inputs)
            maxpool_l = MaxPooling1D(pool_size=int(conv_l.shape[1]),
                                     padding='valid')(conv_l)
            avepool_l = AveragePooling1D(pool_size=int(conv_l.shape[1]),
                                         padding='valid')(conv_l)
            convs.append(maxpool_l)
            convs.append(avepool_l)

        concatenated_tensor = Concatenate(axis=1)(convs)
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(rate=dropout)(flatten)
        dense = Dense(128, activation='softplus')(dropout)
        outputs = Dense(output_dim)(dense)

        model = Model(inputs=inputs, outputs=outputs, name='CNN')
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss='mse', metrics=['mse', 'mae'])
        model.summary()
        self._is_init = True
        self._model = model

    def fit(self, x_train, y_train, x_dev, y_dev, train_loop_num, **kwargs):
        callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3)]
        self._model.fit(
            x_train, y_train,
            epochs=NUM_EPOCH,
            callbacks=callbacks,
            validation_data=(x_dev, y_dev),
            verbose=1,  # Logs once per epoch.
            batch_size=32,
            shuffle=True)

    def predict(self, x_test):
        return self._model.predict(x_test, batch_size=32)


class CnnMulInputs(MyClassifier):
    def __init__(self):
        self._model = None
        self._is_init = False
        self._model = None

    def preprocess_data(self, x):
        return x

    @property
    def is_init(self):
        return self._is_init

    def init_model(self,
                   prompt,
                   input_shape1,
                   input_shape2,
                   output_dim,
                   dropout=0.0,
                   **kwargs):

        embedding_input = Input(shape=(input_shape1[0], input_shape1[1]))
        stringkernel_input = Input(shape=(input_shape2,))
        kernel_sizes = [2, 3, 4]
        convs = []
        for i in range(len(kernel_sizes)):
            conv_l = Conv1D(filters=100,
                            kernel_size=(kernel_sizes[i]),
                            kernel_initializer='normal',
                            bias_initializer='random_uniform',
                            activation='relu',
                            padding='same')(embedding_input)
            maxpool_l = MaxPooling1D(pool_size=int(conv_l.shape[1]),
                                     padding='valid')(conv_l)
            avepool_l = AveragePooling1D(pool_size=int(conv_l.shape[1]),
                                         padding='valid')(conv_l)
            convs.append(maxpool_l)
            convs.append(avepool_l)

        concatenated_tensor = Concatenate(axis=1)(convs)
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(rate=dropout)(flatten)

        x = Concatenate(axis=-1)([dropout, stringkernel_input])

        x = Dense(512, activation='softplus')(x)
        x = Dense(512, activation='softplus')(x)
        x = Dense(512, activation='softplus')(x)
        x = Dense(512, activation='softplus')(x)
        x = Dense(512, activation='softplus')(x)
        outputs = Dense(output_dim)(x)

        model = Model(inputs=[embedding_input, stringkernel_input], outputs=outputs, name='CNN')
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss='mse', metrics=['mse', 'mae'])
        model.summary()
        self._is_init = True
        self._model = model

    def fit(self, x_train, y_train, x_dev, y_dev, train_loop_num, **kwargs):
        callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3)]
        self._model.fit(
            x_train, y_train,
            epochs=NUM_EPOCH,
            callbacks=callbacks,
            validation_data=(x_dev, y_dev),
            verbose=1,  # Logs once per epoch.
            batch_size=32,
            shuffle=True)

    def predict(self, x_test):
        return self._model.predict(x_test, batch_size=32)