import json
import os

from bias_assessment_module.supplier.ModelAndCorpusSupplier import ModelAndCorpusSupplier
from bias_assessment_module.supplier.impl.CLLIPSupplier import CLLIPSupplier
from bias_assessment_module.supplier.impl.CPBCSupplier import CPBCSupplier
from bias_assessment_module.supplier.impl.ChiLitSupplier import ChiLitSupplier
from bias_assessment_module.supplier.impl.GAPSupplier import GAPSupplier
from bias_assessment_module.supplier.impl.GoogleNewsSupplier import GoogleNewsSupplier


class ModelSupplierFactory:
    """
    A class that provides concrete implementations of the ModelSupplier class based on the model configuration.
    """

    model_types = {
        "ChiLit": ChiLitSupplier,
        "CLLIP": CLLIPSupplier,
        "CPBC": CPBCSupplier,
        "GoogleNews": GoogleNewsSupplier,
        "GAP": GAPSupplier
    }


    @staticmethod
    def create_model_supplier(model_config):
        """
        instantiates a model object specified in the corpus-specific configuration file _config.json.
        :param model_config: an object of type ModelConfig
        :return: new ModelSupplier object
        """
        config_path = open(os.path.join(model_config.corpus_path, ModelAndCorpusSupplier.CONFIGURATION_FILENAME), "r")
        corpus_config = json.load(config_path)
        model_type = corpus_config["model_type"]
        model_type_class = ModelSupplierFactory.model_types[model_type]
        return model_type_class(model_config.corpus_path, corpus_config, model_config)



