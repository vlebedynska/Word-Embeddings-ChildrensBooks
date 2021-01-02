import os
from abc import ABC, abstractmethod

from bias_assessment_module.ModelSupplier import ModelSupplier

class CorpusSupplier(ABC):
    """
    :param test test

    """

    @abstractmethod
    def _load_data(self):
        """
        Method which loads data
        :returns list of corpora in form of tokens list
        """
        pass