import json
import os

from CorpusSupplier import CorpusSupplier
from CorpusSupplierCLLIC import CorpusSupplierCLLIC
from CorpusSupplierChiLit import CorpusSupplierChiLit
from CorpusSupplierChiLitSmall import CorpusSupplierChiLitSmall


class CorpusSupplierFactory:

    corpus_types = {
        "CorpusSupplierChiLit": CorpusSupplierChiLit,
        "CorpusSupplierCLLIC": CorpusSupplierCLLIC,
        "CorpusSupplierChiLitSmall": CorpusSupplierChiLitSmall
    }


    @staticmethod
    def create_corpus_supplier(corpus_path):
        config_path = open(os.path.join(corpus_path, CorpusSupplier.CONFIGURATION_FILENAME), "r")
        config = json.load(config_path)
        corpus_type = config["corpus_type"]
        corpus_type_class = CorpusSupplierFactory.corpus_types[corpus_type]
        return corpus_type_class(corpus_path)

