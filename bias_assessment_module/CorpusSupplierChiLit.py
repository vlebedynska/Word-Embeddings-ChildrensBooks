from abc import ABC

import gensim

from bias_assessment_module.CorpusSupplier import CorpusSupplier


class CorpusSupplierChiLit(CorpusSupplier, ABC):

    def __init__(self, corpus_path, config):
        super().__init__(corpus_path, config)

    def load_data(self):
        output_text = []
        for file_name in CorpusSupplier.get_files(self):
            with open(file_name, 'r') as file:
                output_text.append(gensim.utils.simple_preprocess(file.read()))
            print("Done appending " + file_name)
        return output_text
