import json
import os

from bias_assessment_module.supplier.ModelAndCorpusSupplier import ModelAndCorpusSupplier
from bias_assessment_module.supplier.impl.CLLIPSupplier import CLLICSupplier
from bias_assessment_module.supplier.impl.CPBCSupplier import CPBCSupplier
from bias_assessment_module.supplier.impl.ChiLitSupplier import ChiLitSupplier
from bias_assessment_module.supplier.impl.GAPSupplier import GAPSupplier
from bias_assessment_module.supplier.impl.GoogleNewsSupplier import GoogleNewsSupplier


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
        config_path = open(os.path.join(model_config.corpus_path, ModelAndCorpusSupplier.CONFIGURATION_FILENAME), "r")
        corpus_config = json.load(config_path)
        model_type = corpus_config["model_type"]
        model_type_class = ModelSupplierFactory.model_types[model_type]
        return model_type_class(model_config.corpus_path, corpus_config, model_config)



