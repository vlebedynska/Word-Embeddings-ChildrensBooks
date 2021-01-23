from abc import ABC, abstractmethod


class ModelSupplier(ABC):
    """
    An abstract class that defines functions for saving and loading the word embedding model.
    """

    @abstractmethod
    def load_models(self):
        pass

    @abstractmethod
    def _load_model(self, model_id):
        pass

    @abstractmethod
    def save_models(self):
        pass

    @abstractmethod
    def _save_model(self, model_id, model):
        pass

    @abstractmethod
    def _config_to_id(self):
        pass

    @property
    def model_id(self):
        return self._config_to_id()
