#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/12/15 21:25
# @Author:  Mecthew
from modeling.models import MyClassifier
import numpy as np
import keras
from keras import optimizers
from keras.layers import (Input, Dense, Dropout, Convolution2D, LSTM, Bidirectional, Flatten, CuDNNLSTM, Concatenate,
                          MaxPooling2D, ELU, Reshape, CuDNNGRU, Conv1D, MaxPooling1D, AveragePooling1D)
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.models import Model
from modeling.constant import MAX_SEQ_LEN, TEXT_DIM


class Crnn(MyClassifier):
    def __init__(self):
        self._model = None
        self._is_init = False

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
            # maxpool_l = MaxPooling1D(pool_size=int(conv_l.shape[1]),
            #                          padding='valid')(conv_l)
            # avepool_l = AveragePooling1D(pool_size=int(conv_l.shape[1]),
            #                              padding='valid')(conv_l)
            # convs.append(maxpool_l)
            # convs.append(avepool_l)
            convs.append(conv_l)

        concatenated_tensor = Concatenate(axis=1)(convs)
        # x = Flatten()(concatenated_tensor)
        x = CuDNNGRU(64, return_sequences=True, name='gru1')(concatenated_tensor)
        x = CuDNNGRU(64, return_sequences=False, name='gru2')(x)
        outputs = Dense(output_dim)(x)

        model = Model(inputs=inputs, outputs=outputs, name='CRNN')
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
            epochs=150,
            callbacks=callbacks,
            validation_data=(x_dev, y_dev),
            verbose=1,  # Logs once per epoch.
            batch_size=32,
            shuffle=True)

    def predict(self, x_test):
        return self._model.predict(x_test, batch_size=32)
