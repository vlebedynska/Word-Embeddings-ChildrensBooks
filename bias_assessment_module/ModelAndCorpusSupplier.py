from glob import glob
import os
from abc import ABC

import gensim
from gensim.models import Word2Vec

from bias_assessment_module.CorpusSupplier import CorpusSupplier
from bias_assessment_module.ModelSupplier import ModelSupplier


class ModelAndCorpusSupplier(ModelSupplier, CorpusSupplier, ABC):

    CONFIGURATION_FILENAME = "_config.json"


    def __init__(self, corpus_path, corpus_config, model_config):
        self._corpus_path = corpus_path
        self._corpus_config = corpus_config
        self._model_config = model_config

    def get_files(self):
        file_to_ignore = os.path.join(self._corpus_path,ModelAndCorpusSupplier.CONFIGURATION_FILENAME)
        for file_name in glob(self._corpus_path+"/*"):
            if file_name == file_to_ignore:
                continue
            yield file_name

    def load_models(self):
        models = []
        corpora_amount = self._model_config["amount_of_corpora"]
        for counter in range(corpora_amount):
            models.append(self._load_model(self._get_model_id_local(counter)))
        return models

    def save_models(self):
        corpora = self._load_data(self._model_config["amount_of_corpora"])
        if not os.path.exists(self._config_to_id()):
            os.makedirs(self._config_to_id())
        for file_name in os.listdir(self._config_to_id()):
            os.remove(os.path.join(self._config_to_id(), file_name))
        for counter, corpus in enumerate(corpora):
            model = gensim.models.Word2Vec(corpus, size=self._model_config["size"],
                                           window=self._model_config["window"],
                                           iter=self._model_config["epochs"], min_count=2, workers=4)
            self._save_model(self._get_model_id_local(counter), model)

    def _get_model_id_local(self, counter):
        return self._config_to_id() + os.path.sep + str(counter)

    def _config_to_id(self):
        return "{model_path}{corpus_name}_amount{amount_of_corpora}_size{size}_wnd{window}_sg{sg}_e{epochs}".format(
            **self._model_config)
