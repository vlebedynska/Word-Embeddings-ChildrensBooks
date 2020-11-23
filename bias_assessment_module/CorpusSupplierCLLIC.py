from abc import ABC
import xml.etree.ElementTree as ET

import gensim
from bias_assessment_module.CorpusSupplier import CorpusSupplier


class CorpusSupplierCLLIC(CorpusSupplier, ABC):

    def __init__(self, corpus_path, config):
        super().__init__(corpus_path, config)

    def load_data(self):
        output_text = []
        for file_name in CorpusSupplier.get_files(self):
            with open(file_name, 'r') as file:
                tree = ET.parse(file)
                root = tree.getroot()
                for i in root.iter("wtext"):
                    xml_output_text = ""
                    for ix in i.findall(".//"):
                        if ix.text != None:
                            xml_output_text += ix.text
                    output_text.append(gensim.utils.simple_preprocess(xml_output_text))
            print("Done appending " + file_name)
        return output_text
