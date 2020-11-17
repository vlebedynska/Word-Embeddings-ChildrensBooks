import os
from abc import ABC

import gensim

from CorpusSupplier import CorpusSupplier


class CorpusSupplierChiLit(CorpusSupplier, ABC):

    def __init__(self, corpus_path):
        super().__init__(corpus_path)

    def load_data(self):
        output_text = []
        for file_name in CorpusSupplier.get_files(self):
            with open(file_name, 'r') as file:
                output_text.append(gensim.utils.simple_preprocess(file.read()))
            print("Done appending " + file_name)
        return output_text
