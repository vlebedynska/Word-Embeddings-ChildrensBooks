import random
from abc import ABC

import gensim
from gensim.models import Word2Vec

from bias_assessment_module.ModelAndCorpusSupplier import ModelAndCorpusSupplier


class ChiLitSupplier(ModelAndCorpusSupplier):

    def __init__(self, corpus_path, corpus_config, model_config):
        super().__init__(corpus_path, corpus_config, model_config)

    def load_models(self):
        models = []
        corpora_amount = self._model_config["amount_of_corpora"]
        for counter in range(corpora_amount):
            model_id_local = self._config_to_id() + "_" + str(counter)
            models.append(Word2Vec.load(model_id_local))
        return models

    def save_models(self):
        corpora = self._load_data(self._model_config["amount_of_corpora"])
        models = []
        for counter, corpus in enumerate(corpora):
            model_id_local = self._config_to_id() + "_" + str(counter)
            model = gensim.models.Word2Vec(corpus, size=self._model_config["size"],
                                           window=self._model_config["window"],
                                           iter=self._model_config["epochs"], min_count=2, workers=4)
            model.save(model_id_local)
            models.append(model)
        return models

    def _load_single_corpus(self):
        output_text = []
        for file_name in sorted(self.get_files(), key=str.lower):
            with open(file_name, 'r') as file:
                file.readline()  # skip line "Title:..."
                file.readline()  # skip line "Author:..."
                output_text.append(gensim.utils.simple_preprocess(file.read()))
                print("Done appending " + file_name)
        corpora = [output_text]
        return corpora



    def _load_multiple_corpora(self):
        my_files = [file_name for file_name in self.get_files()]
        random.shuffle(my_files)
        for file_name in my_files:
            print("Start appending " + file_name)
            with open(file_name, 'r') as file:
                file.readline()  # skip line "Title:..."
                file.readline()  # skip line "Author:..."
                for line in file:
                    tokens = gensim.utils.simple_preprocess(line)
                    yield tokens, False
                yield [], True # document end reached


    def _config_to_id(self):
        return "{model_path}{corpus_name}_size{size}_wnd{window}_sg{sg}_e{epochs}".format(**self._model_config)
