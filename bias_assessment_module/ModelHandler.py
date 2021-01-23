import os

from bias_assessment_module.supplier.ModelSupplierFactory import ModelSupplierFactory


class ModelHandler:
    """
    A class that encapsulates the mechanism which provides FastText or word2vec model instance.
    """

    def __init__(self, model_config, force=False):
        self._model_id = -1
        self._model_config = model_config
        self._models = self._load(force)

    @staticmethod
    def create_and_load(model_config, force_training=False):
        """
        creates an instance of the ModelHandler class.
        :param model_config: model configuration
        :param force_training: forces retraining, even if the model is cached
        :return: ModelHandler
        """
        model_handler = ModelHandler(model_config, force_training)
        return model_handler

    def _load(self, force_training=False):
        """
        loads the word embedding models. If the models do not exist, they are created and cached in the file system.
        After that, the models are loaded from the cache.
        :param force_training: forces retraining, even if the model is cached
        :return: List of models
        """
        model_supplier = ModelSupplierFactory.create_model_supplier(self._model_config)
        self._model_id = model_supplier.model_id
        if not os.path.exists(self._model_id) or force_training:
            model_supplier.save_models()
        print("Load model")
        return model_supplier.load_models()


    @property
    def models(self):
        return self._models

    @property
    def model_id(self):
        return self._model_id
