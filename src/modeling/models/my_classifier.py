import abc


class MyClassifier(object):
    @property
    @abc.abstractmethod
    def is_init(self):
        pass

    @abc.abstractmethod
    def init_model(self, **kwargs):
        pass

    @abc.abstractmethod
    def preprocess_data(self, x):
        pass

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, x_test):
        pass
