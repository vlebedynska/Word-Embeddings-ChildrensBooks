import os
import gensim

from CorpusSupplier import CorpusSupplier


class CorpusSupplierChiLit(CorpusSupplier):

    def __init__(self, corpus_path):
        self._corpus_path = corpus_path

    def load_data(self):
        output_text = []
        for file_name in os.listdir(self._corpus_path):
            with open(os.path.join(self._corpus_path, file_name), 'r') as file:
                for line in file:
                    output_text.append(gensim.utils.simple_preprocess(line))
            print("Done appending " + file_name)
        return output_text
