import os

import gensim


class CorpusSupplier:
    def __init__(self):
        raise NotImplementedError('CorpusSupplier is an abstract class!')

    def load_data(self):
        raise NotImplementedError('subclasses must override load_data()!')
        # output_text = []
        # for file_name in os.listdir(self._corpus_path):
        #     with open(os.path.join(self._corpus_path, file_name), 'r') as file:
        #         for line in file:
        #             output_text.append(gensim.utils.simple_preprocess(line))
        #     print("Done appending " + file_name)
        # return output_text
