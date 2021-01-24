import re
from abc import ABC


import gensim
from gensim.models import Word2Vec

from bias_assessment_module.supplier.ModelAndCorpusSupplier import ModelAndCorpusSupplier


class CPBCSupplier(ModelAndCorpusSupplier):
    """
    A class that provides concrete implementation of the ModelSupplier and CorpusSupplier,
    implements the concrete functions for loading and saving of the CPB Corpus.
    """

    def __init__(self, corpus_path, corpus_config, model_config):
        super().__init__(corpus_path, corpus_config, model_config)

    def _save_model(self, model_id, model):
        """
        saves the model.
        :param model_id: model id, used as path to the file where the model is to be stored
        :param model: model
        """
        return model.save(model_id)

    def _load_model(self, model_id):
        """
        loads a word2vec model by the model id.
        :param model_id: model id
        :return: word2vec model
        """
        return Word2Vec.load(model_id)

    def _load_multiple_corpora(self):
        """
        raises an error if the multiple corpora are to be loaded.
        :return: new ValueError instance
        """
        raise ValueError(self._model_config["corpus_name"] + " is too short to be partitioned in several corpora.")

    def _load_single_corpus(self):
        """
        loads a single corpus from the file system and removes lines that specify the title of the book and the author.
        :return: list of sentences of a corpus, where a sentence represents all books in a corpus
        """
        output_text = []
        for file_name in ModelAndCorpusSupplier.get_files(self):
            with open(file_name, 'r') as file:
                pattern = re.compile('(^|\n)(Title:|Author:).*?\n') # find lines that specify the title of the book and the author
                text_without_authors_titles = pattern.sub("", file.read()) # apply the pattern
                output_text.append(gensim.utils.simple_preprocess(text_without_authors_titles))
            print("Done appending " + file_name)
        corpora = [output_text]
        return corpora
