import random
from abc import ABC
import xml.etree.ElementTree as ET

import gensim
from gensim.models import Word2Vec

from bias_assessment_module.supplier.ModelAndCorpusSupplier import ModelAndCorpusSupplier


class CLLIPSupplier(ModelAndCorpusSupplier):

    def __init__(self, corpus_path, corpus_config, model_config):
        super().__init__(corpus_path, corpus_config, model_config)

    def _load_multiple_corpora(self):
        my_files = [file_name for file_name in self.get_files()]
        random.shuffle(my_files)
        for file_name in my_files:
            print("Start appending " + file_name)
            with open(file_name, 'r') as file:
                tree = ET.parse(file)
                root = tree.getroot()
                for elements in root.iter("wtext"):
                    xml_output_text = ""
                    for element in elements.findall(".//"):
                        if element.text is not None:
                            xml_output_text += element.text
                        if element.text == "\n":
                            tokens = gensim.utils.simple_preprocess(xml_output_text)
                            xml_output_text = ""
                            yield tokens, False
                yield [], True  # document end reached

    def _load_single_corpus(self):
        output_text = []
        for file_name in ModelAndCorpusSupplier.get_files(self):
            with open(file_name, 'r') as file:
                tree = ET.parse(file)
                root = tree.getroot()
                for i in root.iter("wtext"):
                    xml_output_text = ""
                    for ix in i.findall(".//"):
                        if ix.text is not None:
                            xml_output_text += ix.text
                    output_text.append(gensim.utils.simple_preprocess(xml_output_text))
            print("Done appending " + file_name)
        corpora = [output_text]
        return corpora

    def _load_model(self, model_id):
        return Word2Vec.load(model_id)

    def _save_model(self, model_id, model):
        return model.save(model_id)