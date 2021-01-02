import os
from abc import ABC

from bias_assessment_module.CorpusSupplier import CorpusSupplier
from bias_assessment_module.ModelSupplier import ModelSupplier


class ModelAndCorpusSupplier(ModelSupplier, CorpusSupplier, ABC):

    CONFIGURATION_FILENAME = "_config.json"

    def __init__(self, corpus_path, corpus_config, model_config):
        self._corpus_path = corpus_path
        self._corpus_config = corpus_config
        self._model_config = model_config


    def get_files(self):
        for file_name in os.listdir(self._corpus_path):
            if file_name == ModelAndCorpusSupplier.CONFIGURATION_FILENAME:
                continue
            yield os.path.join(self._corpus_path, file_name)

    @property
    def model_id(self):
        return self._config_to_id()