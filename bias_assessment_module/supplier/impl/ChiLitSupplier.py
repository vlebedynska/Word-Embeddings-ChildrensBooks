import random

import gensim
from gensim.models import Word2Vec

from bias_assessment_module.supplier.ModelAndCorpusSupplier import ModelAndCorpusSupplier


class ChiLitSupplier(ModelAndCorpusSupplier):
    """
    A class that provides concrete implementation of the ModelSupplier and CorpusSupplier,
    implements the concrete functions for loading and saving of the ChiLit Corpus.
    """

    def __init__(self, corpus_path, corpus_config, model_config):
        super().__init__(corpus_path, corpus_config, model_config)

    def _load_model(self, model_id):
        """
        loads a word2vec model by the model id.
        :param model_id: model id
        :return: word2vec model
        """
        return Word2Vec.load(model_id)

    def _save_model(self, model_id, model):
        """
        saves the model.
        :param model_id: model id, used as path to the file where the model is to be stored
        :param model: model
        """
        model.save(model_id)

    def _load_single_corpus(self):
        """
        loads a single corpus from the file system and removes lines that specify the title of the book and the author.
        :return: list of sentences of a corpus, where a sentence represents a book
        """
        output_text = []
        for file_name in sorted(self.get_files(), key=str.lower):
            with open(file_name, 'r') as file:
                file.readline()  # skip line "Title:..."
                file.readline()  # skip line "Author:..."
                output_text.append(gensim.utils.simple_preprocess(file.read()))
                print("Done appending " + file_name)
        sentences = [output_text]
        return sentences

    def _load_multiple_corpora(self):
        """
        loads multiple corpora from the file system and removes lines that specify the title of the book and the author.
        :return: yields corpus data line by line
        """
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
