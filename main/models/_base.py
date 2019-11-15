#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):

    @property
    @abstractmethod
    def type(self):
        """Define model type."""
        pass

    @staticmethod
    @abstractmethod
    def get_model():
        """Get model instance."""
        pass

    @abstractmethod
    def predict(self, x):
        """Return predicted result."""
        pass

    def __call__(self, x):
        """Return predicted result."""
        return self.predict(x)

    def set_weights_by_config(self):
        """Set weights from the weights saved at the pass stored at Config."""
        path = Config.MODELS.get(self.type)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError('Could not find weights. Path is not set or Model is not implemented yet')

        self._model.load_weights(path)

    def set_weights(self, path):
        """Set weights from the weights saved at the given path."""
        self._model.load_weights(path)

    @abstractmethod
    def _build_model(self):
        pass
