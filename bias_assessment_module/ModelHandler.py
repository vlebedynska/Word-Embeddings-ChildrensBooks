import os
import gensim
from gensim.models import Word2Vec, fasttext, KeyedVectors

from bias_assessment_module.CorpusSupplierFactory import CorpusSupplierFactory


class ModelHandler:

    def __init__(self, model_config, force=False):
        self._config = model_config
        self._model_id = self._config_to_id()
        self._model = self._load(force)

    @staticmethod
    def create_and_load(model_config, force_training=False):
        model_handler = ModelHandler(model_config, force_training)
        return model_handler

    def _load(self, force_training=False):
        model_type = self._config["model_type"]
        if model_type == "w2v":
            if not os.path.exists(self._model_id) or force_training:
                return self._save(self._model_id)
            else:
                print("Load model")
                return KeyedVectors.load_word2vec_format(self._model_id, binary=True)
                # return Word2Vec.load(self._model_id)
        elif model_type == "ft":
            return gensim.models.fasttext.load_facebook_vectors(self._model_id)
        else:
            raise ValueError("Unsupported Word Embedding type '{}'".format(model_type))

    def _save(self, model_id):
        documents = CorpusSupplierFactory.create_corpus_supplier(self._config["corpus_path"]).load_data()
        model = gensim.models.Word2Vec(documents, size=self._config["size"], window=self._config["window"], min_count=2, workers=4)
        model.train(documents, total_examples=len(documents), epochs=self._config["epochs"])
        model.save(model_id)
        return model

    def _config_to_id(self):
        return "{model_path}{corpus_name}_size{size}_wnd{window}_sg{sg}_e{epochs}".format(**self._config)


    @property
    def model(self):
        return self._model

    @property
    def model_id(self):
        return self._model_id
