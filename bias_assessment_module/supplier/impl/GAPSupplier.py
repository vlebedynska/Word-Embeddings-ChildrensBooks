import os
from shutil import copyfile

import gensim

from bias_assessment_module.supplier.ModelSupplier import ModelSupplier


class GAPSupplier(ModelSupplier):
    """
    A class that implements the concrete functions for loading the model of the GAP corpus and copying it to the cache.
    """

    def __init__(self, corpus_path, corpus_config, model_config):
        self._corpus_path = corpus_path
        self._corpus_config = corpus_config
        self._model_config = model_config

    def load_models(self):
        """
        loads FastText model of the GAP corpus.
        :return: 1-element list (of models) consisting of the single GAP model
        """
        return [self._load_model(self._config_to_id())]

    def save_models(self):
        """
        copies the GAP model from from its storage location to the cache.
        :return: None
        """
        copyfile(self._corpus_path + os.path.sep + self._corpus_config["model_file"], self._config_to_id())

    def _load_model(self, model_id):
        """
        loads the FastText model by the model id, used as path to the file where the model is stored.
        :param model_id: model id
        :return: FastText model of the GAP corpus
        """
        return gensim.models.fasttext.load_facebook_vectors(model_id)

    def _save_model(self, model_id, model):
        pass

    def _config_to_id(self):
        """
        creates a unique id for the directory where the model is stored,
        serves as the path to the directory under which the model is stored.
        :return: directory name
        """
        return "{0.model_path}".format(self._model_config)