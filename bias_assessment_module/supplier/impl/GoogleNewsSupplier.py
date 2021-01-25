import os
from shutil import copyfile

from gensim.models import KeyedVectors

from bias_assessment_module.supplier.ModelSupplier import ModelSupplier


class GoogleNewsSupplier(ModelSupplier):
    """
    A class that implements the concrete functions for loading the model of the GoogleNews corpus and copying it to the cache.
    """

    def __init__(self, corpus_path, corpus_config, model_config):
        self._corpus_path = corpus_path
        self._corpus_config = corpus_config
        self._model_config = model_config

    def load_models(self):
        """
        loads the word2vec model of the GoogleNews corpus.
        :return: 1-element/sized list consisting of the single GoogleNews model
        """
        return [self._load_model(self._config_to_id())]

    def _load_model(self, model_id):
        """
        loads model by the model id, used as path to the file where the model is stored.
        :param model_id: model id
        :return: model of the GoogleNews corpus
        """
        return KeyedVectors.load_word2vec_format(model_id, binary=True)

    def save_models(self):
        """
        copies the GoogleNews model from from the original storage location to the cache.
        :return: None
        """
        copyfile(self._corpus_path + os.path.sep + self._corpus_config["model_file"], self._config_to_id())

    def _save_model(self, model_id, model):
        pass #TODO

    def _config_to_id(self):
        """
        creates a unique id for the directory where the model is stored,
        serves as the path to the directory under which the model is stored.
        :return: directory name
        """
        return "{0.model_path}.bin.gz".format(self._model_config)

