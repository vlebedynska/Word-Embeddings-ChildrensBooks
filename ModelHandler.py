import os
import gensim
from gensim.models import Word2Vec

from CorpusSupplierFactory import CorpusSupplierFactory


class ModelHandler:
    def __init__(self, model_name, corpus_path, model_type, force=False):
        self._model = self.load(model_name, corpus_path, model_type)

    @staticmethod
    def create_and_load(model_name, corpus_path, model_type, force=False):
        path_to_database = "database.txt"
        if os.path.isfile(path_to_database):
            with open("database.txt", "r") as database:
                str2 = corpus_path + "\t" + model_type + "\n"
                if str2 in database.readlines():
                    print("True")
        model_handler = ModelHandler(model_name, corpus_path, model_type, force)
        return model_handler

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
        documents = CorpusSupplierFactory.create_corpus_supplier(corpus_path).load_data()
        model = gensim.models.Word2Vec(documents, size=150, window=10, min_count=2, workers=4)
        model.train(documents, total_examples=len(documents), epochs=10)
        model.save("data/cache/"+model_name)
        file = open("database.txt", "a")
        file.write(corpus_path + "\t" + "w2v" + model_name + "\n")
        file.close()
        return model

    @property
    def model(self):
        return self._model
