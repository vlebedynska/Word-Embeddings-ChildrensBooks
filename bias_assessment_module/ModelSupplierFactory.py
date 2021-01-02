import json
import os

from bias_assessment_module.CLLICSupplier import CLLICSupplier
from bias_assessment_module.CPBCSupplier import CPBCSupplier
from bias_assessment_module.ChiLitSupplier import ChiLitSupplier
from bias_assessment_module.GAPSupplier import GAPSupplier
from bias_assessment_module.GoogleNewsSupplier import GoogleNewsSupplier
from bias_assessment_module.ModelAndCorpusSupplier import ModelAndCorpusSupplier


class ModelSupplierFactory:

    model_types = {
        "ChiLit": ChiLitSupplier,
        "CLLIC": CLLICSupplier,
        "CPBC": CPBCSupplier,
        "GoogleNews": GoogleNewsSupplier,
        "GAP": GAPSupplier
    }


    @staticmethod
    def create_model_supplier(model_config):
        config_path = open(os.path.join(model_config["corpus_path"], ModelAndCorpusSupplier.CONFIGURATION_FILENAME), "r")
        corpus_config = json.load(config_path)
        model_type = corpus_config["model_type"]
        model_type_class = ModelSupplierFactory.model_types[model_type]
        return model_type_class(model_config["corpus_path"], corpus_config, model_config)



