import gensim

from bias_assessment_module.ModelSupplier import ModelSupplier


class GAPSupplier(ModelSupplier):

    def __init__(self, corpus_path, config):
        super().__init__(corpus_path, config)

    def load_models(self):
        if self._config["amount_of_corpora"] > 1:
            print(model_type + " model type does not support multiple amount of corpora. "
                               "Only single model will be loaded.")
        return gensim.models.fasttext.load_facebook_vectors(self._model_id)

    def save_models(self):
        raise ValueError("Model saving is not supported.")

    def _config_to_id(self):
        pass