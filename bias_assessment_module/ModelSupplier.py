from abc import ABC, abstractmethod


class ModelSupplier(ABC):

    @abstractmethod
    def load_models(self):
        pass

    @abstractmethod
    def save_models(self):
        pass

    @abstractmethod
    def _config_to_id(self):
        pass

