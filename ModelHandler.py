import os
import gensim
import json
from gensim.models import Word2Vec

import CorpusSupplierChiLit
import CorpusSupplierCLLIC


class ModelHandler:
    corpus_types = {
        "CorpusSupplierChiLit" : CorpusSupplierChiLit.CorpusSupplierChiLit,
        "CorpusSupplierCLLIC" : CorpusSupplierCLLIC.CorpusSupplierCLLIC
    }

    def __init__(self, model_name, corpus_path, model_type, force=False):
        self._model = self.load(model_name, corpus_path, model_type)

    def load(self, model_name, corpus_path, model_type, force=False):
        if model_type == "w2v":
            if not os.path.exists(model_name) or force:
                return self.save(model_name, corpus_path)
            else:
                print("Load model")
                return Word2Vec.load(model_name)
        elif model_type == "ft":
            return gensim.models.fasttext.load_facebook_vectors(model_name)
        else:
            raise ValueError("Unsupported Word Embedding type '{}'".format(model_type))

    def save(self, model_name, corpus_path):
        documents = ModelHandler.create_corpus_supplier(corpus_path).load_data()
        # output_text = []
        # for document in documents:
        #     output_text.append(gensim.utils.simple_preprocess(document))
        # print(output_text[0])
        model = gensim.models.Word2Vec(documents, size=150, window=10, min_count=2, workers=4)
        model.train(documents, total_examples=len(documents), epochs=10)
        model.save(model_name)
        return model

    @staticmethod
    def create_corpus_supplier(corpus_path):
        config_path = open(os.path.join(corpus_path, "_config.json"), "r")
        config = json.load(config_path)
        corpus_type = config["corpus_type"]
        corpus_type_class = ModelHandler.corpus_types[corpus_type]
        return corpus_type_class(corpus_path)

    @property
    def model(self):
        return self._model
