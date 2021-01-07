import os
import gensim
from gensim.models import Word2Vec, fasttext, KeyedVectors

from bias_assessment_module.ModelSupplierFactory import ModelSupplierFactory


class ModelHandler:
    def __init__(self, model_config, force=False):
        self._model_id = -1
        self._config = model_config
        self._models = self._load(force)

    @staticmethod
    def create_and_load(model_config, force_training=False):
        model_handler = ModelHandler(model_config, force_training)
        return model_handler

    def _load(self, force_training=False):
        model_supplier = ModelSupplierFactory.create_model_supplier(self._config)
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
