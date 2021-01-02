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
        corpora = self._load_data()
        models = []
        for counter, corpus in enumerate(corpora):
            model_id_local = self._config_to_id() + "_" + str(counter)
            model = gensim.models.Word2Vec(corpus, size=self._model_config["size"], window=self._model_config["window"],
                                           min_count=2, workers=4)
            model.train(corpus, total_examples=len(corpus), epochs=self._model_config["epochs"])
            model.save(model_id_local)
            models.append(model)
        return models

    def _load_data(self):
        corpora_amount = self._model_config["amount_of_corpora"]
        corpora = []
        for corpus in range(corpora_amount):
            output_text = []
            for file_name in ModelAndCorpusSupplier.get_files(self):
                with open(file_name, 'r') as file:
                    file.readline()  # skip line "Title:..."
                    file.readline()  # skip line "Author:..."
                    text_without_authors_titles = file.read()
                    output_text.append(gensim.utils.simple_preprocess(text_without_authors_titles))
                print("Done appending " + file_name)
            corpora.append(output_text)
        return corpora

    def _config_to_id(self):
        return "{model_path}{corpus_name}_size{size}_wnd{window}_sg{sg}_e{epochs}".format(**self._model_config)



