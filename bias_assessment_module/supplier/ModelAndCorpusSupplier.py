from glob import glob
import os
from abc import ABC

import gensim
from gensim.models import Word2Vec

from bias_assessment_module.supplier.ModelSupplier import ModelSupplier
from bias_assessment_module.supplier.CorpusSupplier import CorpusSupplier


class ModelAndCorpusSupplier(ModelSupplier, CorpusSupplier, ABC):
    """
    A class that implements CorpusSupplier and ModelSupplier abstract
    classes and provides concrete implementation for training the
    word2vec models. It saves and loads models using concrete implementations from the corresponding classes.
    """
    CONFIGURATION_FILENAME = "_config.json"

    def __init__(self, corpus_path, corpus_config, model_config):
        self._corpus_path = corpus_path
        self._corpus_config = corpus_config
        self._model_config = model_config

    def get_files(self):
        """
        gets all files from the directory with the corpus data.
        :return: yields filenames
        """
        file_to_ignore = os.path.join(self._corpus_path, ModelAndCorpusSupplier.CONFIGURATION_FILENAME)
        for file_name in glob(self._corpus_path + "/**", recursive=True):
            if os.path.isdir(file_name) or file_name == file_to_ignore:
                continue
            yield file_name

    def load_models(self):
        """
        loads all models saved in the model directory.
        :return: list of models
        """
        models = []
        corpora_amount = self._model_config.amount_of_corpora
        for counter in range(corpora_amount):
            models.append(self._load_model(self._get_model_id_local(counter)))
        return models

    def save_models(self):
        """
        loads corpus data and creates the number of word2vec models specified in the ModelConfig object
        :return: None
        """
        corpora = self._load_data(self._model_config.amount_of_corpora)
        # if needed, create a directory where the models are stored
        if not os.path.exists(self._config_to_id()):
            os.makedirs(self._config_to_id())
        # remove old cached files
        for file_name in os.listdir(self._config_to_id()):
            os.remove(os.path.join(self._config_to_id(), file_name))
        # for each corpus create a model and save the model in the model directory
        for counter, corpus in enumerate(corpora):
            model = gensim.models.Word2Vec(corpus, size=self._model_config.size,
                                           window=self._model_config.window,
                                           iter=self._model_config.epochs, min_count=2, workers=4)
            self._save_model(self._get_model_id_local(counter), model)

    def _get_model_id_local(self, counter):
        """
        generates the model id as directory name (see _config_to_id()) and counter as the filename
        :param counter: model number
        :return: model id
        """
        return self._config_to_id() + os.path.sep + str(counter)

    def _config_to_id(self):
        """
        creates a unique id for the directory where the models are stored,
        serves as the path to the directory under which the models are stored.
        :return: directory name
        """
        return "{0.model_path}{0.corpus_name}_amount{0.amount_of_corpora}_size{0.size}_wnd{0.window}_sg{0.sg}_e{0.epochs}".format(
            self._model_config)
