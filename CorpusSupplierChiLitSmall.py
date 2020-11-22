import os
from abc import ABC

import gensim
from pandas import np

from CorpusSupplier import CorpusSupplier


class CorpusSupplierChiLitSmall(CorpusSupplier, ABC):

    def __init__(self, corpus_path):
        super().__init__(corpus_path)


    def load_data(self):
        output_text = []
        words_count = 0
        for file_name in CorpusSupplier.get_files(self):
            with open(file_name, 'r') as file:
                data = file.read()
                words_boundary = min(20000, len(data.split()))
                output_text.append(gensim.utils.simple_preprocess(data)[0:words_boundary])
                words_count += words_boundary
            if words_count > 1000000:
                print("Done appending " + file_name)
                return output_text