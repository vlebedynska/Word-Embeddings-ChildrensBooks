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
            return model_supplier.save_models()
        else:
            print("Load model")
            return model_supplier.load_models()


    # def _save(self, model_id, corpus_supplier):
    #     corpora = corpus_supplier.load_data()
    #     models = []
    #     for counter, corpus in enumerate(corpora):
    #         model = gensim.models.Word2Vec(corpus, size=self._config["size"], window=self._config["window"], min_count=2, workers=4)
    #         model.train(corpus, total_examples=len(corpus), epochs=self._config["epochs"])
    #         model.save(model_id+"_"+counter)
    #         models.append(model)
    #     return models
    #
    # def _load_corpora(self, corpus_supplier):
    #     models = []
    #     if corpus_supplier.
    #     corpora_amount = self._config["amount_of_corpora"]
    #     for counter in range(corpora_amount):
    #         models.append(Word2Vec.load(self._model_id+"_"+str(counter)))
    #     # return KeyedVectors.load_word2vec_format(self._model_id, binary=True)
    #     return models

    # def _config_to_id(self):
    #     return "{model_path}{corpus_name}_size{size}_wnd{window}_sg{sg}_e{epochs}".format(**self._config)


    @property
    def models(self):
        return self._models

    @property
    def model_id(self):
        return self._model_id
