import random

import gensim
from gensim.models import Word2Vec

from bias_assessment_module.supplier.ModelAndCorpusSupplier import ModelAndCorpusSupplier


class ChiLitSupplier(ModelAndCorpusSupplier):

    def __init__(self, corpus_path, corpus_config, model_config):
        super().__init__(corpus_path, corpus_config, model_config)

    def _load_model(self, model_id):
        return Word2Vec.load(model_id)

    def _save_model(self, model_id, model):
        return model.save(model_id)

    def _load_single_corpus(self):
        output_text = []
        for file_name in sorted(self.get_files(), key=str.lower):
            with open(file_name, 'r') as file:
                file.readline()  # skip line "Title:..."
                file.readline()  # skip line "Author:..."
                output_text.append(gensim.utils.simple_preprocess(file.read()))
                print("Done appending " + file_name)
        corpora = [output_text]
        return corpora

    def _load_multiple_corpora(self):
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
