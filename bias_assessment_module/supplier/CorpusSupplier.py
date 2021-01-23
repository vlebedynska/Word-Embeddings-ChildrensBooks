import os
from abc import ABC, abstractmethod


class CorpusSupplier(ABC):
    """
    :param get_custom_size_corpus get_custom_size_corpus

    """
    CORPUS_SIZE = 65008

    def _load_data(self, corpora_amount):
        """
        Method which loads data
        :returns list of corpora in form of tokens list
        """
        if corpora_amount == 1:
            return self._load_single_corpus()
        else:
            return self.generate_multiple_corpora_from_raw_data(CorpusSupplier.CORPUS_SIZE, corpora_amount)

    def generate_multiple_corpora_from_raw_data(self, corpus_size, corpora_amount):
        corpora = []
        output_text = []
        output_text_local = []
        current_corpus_size = 0
        for tokens, end_of_document in self._load_multiple_corpora():
            output_text_local.extend(tokens)
            current_corpus_size += len(tokens)
            if end_of_document:
                output_text.append(output_text_local)
                output_text_local = []
                print(" Appending finished. Current corpus size " + str(current_corpus_size))
            if current_corpus_size >= corpus_size:
                if not end_of_document:
                    output_text.append(output_text_local)
                corpora.append(output_text)
                print("Corpus filled: " + str(len(corpora)) + "\t Corpus size: " + str(current_corpus_size))
                output_text = []
                output_text_local = []
                current_corpus_size = 0
            if len(corpora) >= corpora_amount:
                break
        if len(corpora) != corpora_amount:
            raise ValueError("End of data reached. Current corpora ammount is " + str(len(corpora)) + " of expected " +
                             str(corpora_amount))
        return corpora

    @abstractmethod
    def _load_multiple_corpora(self):
        pass

    @abstractmethod
    def _load_single_corpus(self):
        pass
