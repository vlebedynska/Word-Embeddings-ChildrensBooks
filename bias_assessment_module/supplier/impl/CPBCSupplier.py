import re
from abc import ABC


import gensim
from gensim.models import Word2Vec

from bias_assessment_module.supplier.ModelAndCorpusSupplier import ModelAndCorpusSupplier


class CPBCSupplier(ModelAndCorpusSupplier):

    def __init__(self, corpus_path, corpus_config, model_config):
        super().__init__(corpus_path, corpus_config, model_config)

    def _save_model(self, model_id, model):
        return model.save(model_id)

    def _load_model(self, model_id):
        return Word2Vec.load(model_id)

    def _load_multiple_corpora(self):
        raise ValueError(self._model_config["corpus_name"] + " is too short to be partitioned in several corpora.")

    def _load_single_corpus(self):
        output_text = []
        for file_name in ModelAndCorpusSupplier.get_files(self):
            with open(file_name, 'r') as file:
                pattern = re.compile('(^|\n)(Title:|Author:).*?\n')
                text_without_authors_titles = pattern.sub("", file.read())
                output_text.append(gensim.utils.simple_preprocess(text_without_authors_titles))
            print("Done appending " + file_name)
        corpora = [output_text]
        return corpora
