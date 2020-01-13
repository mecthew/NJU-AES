import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import os
import numpy as np

from modeling.model_manager import ModelManager
from integration.tools import log, timeit, label_scalar
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from modeling.constant import NGRAM_MAX_LENGTH, NGRAM_MIN_LENGTH, ROOT_DIR


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


class Model(object):
    def __init__(self, prompt, max_iter=10):
        """ Initialization for model
        """
        self.done_training = False
        self._train_loop_num = 0
        self._model_manager = None
        self._prompt = prompt
        self._max_iter = max_iter
        self._train_features_origin = None
        self._x_scaler = None
        self._y_scaler = None
        self._train_y_min = None
        self._train_y_max = None

    def get_prompt_str(self, prompt, data_type):
        return "prompt@" + str(prompt) + '-' + data_type + '.txt'

    def minmax_scaler(self, y, data_type):
        if data_type == 'train':
            self._train_y_min = np.min(y)
            self._train_y_max = np.max(y)

        y_scaler = [((x - self._train_y_min) * 1.0 / (self._train_y_max - self._train_y_min)) for x in y]
        y_scaler = np.array(y_scaler)

        log(f"y_scaler_{data_type}_shape: {y_scaler.shape}")
        log(f"y_scaler_{data_type}:\n{y_scaler}")

        return y_scaler

    def inverse_scaler(self, y_scaler, data_type):
        y = [(x * 1.0 * (self._train_y_max - self._train_y_min) + self._train_y_min) for x in y_scaler]
        y = np.array(y)
        log(f"y_inverse_{data_type}_shape: {y.shape}")
        log(f"y_inverse_{data_type}:\n{y}")
        return y

    def feature_scaler(self, X):
        scaler = StandardScaler()
        # scaler = MinMaxScaler()
        scaler.fit(X)
        X_scaler = scaler.transform(X)
        return scaler, X_scaler

    def kernel_process_data(self, prompt, data_type):
        hisk_dir = ROOT_DIR + '/essay_data/HISK/output'
        train_output = os.path.join(hisk_dir, self.get_prompt_str(prompt=prompt, data_type=data_type))
        features = None
        with open(train_output, 'r') as file:
            for line in file:
                line_split = line.split()
                features_i = []
                for value in line_split:
                    features_i.append(int(value))
                features_i = np.array(features_i).reshape(1, -1)
                if features is None:
                    features = features_i
                else:
                    # features = np.c_[features, features_i]
                    features = np.concatenate((features, features_i), axis=0)
        return features

    @timeit
    def train(self, train_dataset, dev_dataset):
        """model training on train_dataset.
        :param train_dataset: tuple, (train_x, train_y)
            train_x: list of essays, each essay is a string.
            train_y: list of integers, referred to labels.

        :param dev_dataset: tuple, (dev_x, dev_y)
        """

        self._train_loop_num += 1

        if self._train_loop_num == 1:
            train_x, train_y = train_dataset
            dev_x, dev_y = dev_dataset
            train_x = self.kernel_process_data(prompt=self._prompt, data_type='train')
            # self._x_scaler, train_x = self.feature_scaler(train_x)

            # log(f"X_scalar_mean_shape: {self._x_scaler.mean_.shape}")
            # log(f"X_scalar_mean: {self._x_scaler.mean_}")
            dev_x = self.kernel_process_data(prompt=self._prompt, data_type='dev')
            # dev_x = self._x_scaler.transform(dev_x)
            # _, dev_x = self.feature_scalar(dev_x)

            #train_y = self.minmax_scaler(train_y, data_type ='train')
            #dev_y = self.minmax_scaler(dev_y, data_type='dev')

            train_dataset = (train_x, train_y)
            dev_dataset = (dev_x, dev_y)

            self._model_manager = ModelManager(train_set=train_dataset, dev_set=dev_dataset)

        self._model_manager.fit(train_loop_num=self._train_loop_num)

        if self._train_loop_num >= self._max_iter:
            self.done_training = True

    @timeit
    def predict(self, x_test, is_dev):
        """
        :param is_dev: true when use dev_dataset.
        :param x_test: list of essays..
        :return: list of integers.
        """

        x_test = self.kernel_process_data(prompt=self._prompt, data_type='test')
        # x_test = self._x_scaler.transform(x_test)
        # _, x_test = self.feature_scalar(x_test)

        log(f"x_test_shape: {x_test.shape}")

        # prediction must be integers
        pred_y, pred_y_dev = self._model_manager.predict(x_test, is_dev=is_dev)

        #pred_y = self.inverse_scaler(pred_y, data_type='pred_y')
        #pred_y_dev = self.inverse_scaler(pred_y_dev, data_type='pred_y_dev')

        log(f"pred_y_shape: {pred_y.shape}")

        log(f'Head 10 of prediction_test_ensemble:\n{pred_y[:10]}')
        log(f'Head 10 of prediction_dev_ensemble:\n{pred_y_dev[:10]}')
        return pred_y, pred_y_dev


if __name__ == '__main__':
    pass
