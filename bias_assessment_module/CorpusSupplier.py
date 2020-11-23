import os
from abc import ABC, abstractmethod
import gensim


class CorpusSupplier(ABC):
    """
    :param test test

    """

    CONFIGURATION_FILENAME = "_config.json"

    def __init__(self, corpus_path, config):
        self._corpus_path = corpus_path
        self._config = config

    @abstractmethod
    def load_data(self):
        """
        Method which loads data
        :returns list of str
        """
        pass

    def get_files(self):
        for file_name in os.listdir(self._corpus_path):
            if file_name == CorpusSupplier.CONFIGURATION_FILENAME:
                continue
            yield os.path.join(self._corpus_path, file_name)
