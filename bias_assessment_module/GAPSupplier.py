from abc import ABC

import gensim

from bias_assessment_module.ModelAndCorpusSupplier import ModelAndCorpusSupplier


class GAPSupplier(ModelAndCorpusSupplier):

    def _load_data(self):
        pass

    def __init__(self, corpus_path, corpus_config, model_config):
        super().__init__(corpus_path, corpus_config, model_config)

    def load_models(self):
        return [gensim.models.fasttext.load_facebook_vectors(self._config_to_id())]

    def save_models(self):
        raise ValueError("Model saving is not supported.")

    def _config_to_id(self):
        return "{model_path}{corpus_name}".format(**self._model_config)