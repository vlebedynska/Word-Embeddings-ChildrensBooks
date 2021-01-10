from bias_assessment_module.BiasAssessor import BiasAssessor
from bias_assessment_module.BiasAssessorException import BiasAssessorException
from bias_assessment_module.EmbeddingsClusterer import EmbeddigsClusterer
from bias_assessment_module.Evaluator import Evaluator
from bias_assessment_module.Logger import Logger
from bias_assessment_module.ModelHandler import ModelHandler
from bias_assessment_module.config.EmbeddingsClustererConfig import EmbeddingsClustererConfig
from bias_assessment_module.config.ModelConfig import ModelConfig
from bias_assessment_module.config.WeatConfig import WeatConfig

LOG_SUFFIX = ".log"
RESULTS_SUFFIX = "_results.txt"
FULL_RESULTS_SUFFIX = "_results_full.txt"
CLUSTERING_RESULTS_SUFFIX = "_results_clusters.txt"

class BiasAssessmentModule():
    def __init__(self, config):
        self._config = config
        self._model_config = ModelConfig(config["model"])
        self._model_clusterer_config = EmbeddingsClustererConfig(config["clustering"])
        self._weat_configs = self.create_weat_configs()
        self._model_handler = ModelHandler.create_and_load(self._model_config)
        self._logger = Logger(self._model_handler.model_id)
        self._bias_assessor = BiasAssessor.create(self._model_handler.models)

    def create_weat_configs(self):
        weat_configs = {}
        for bias_category  in self._config["weat_lists"]:
            weat_config_json = self._config["weat_lists"][bias_category]
            weat_config_json_extracted = {}
            for entry in weat_config_json:
                for list_name in weat_config_json[entry]:
                    weat_config_json_extracted.update({list_name: weat_config_json[entry][list_name]})
            weat_config_json_extracted["name"] = bias_category
            weat_config_object = WeatConfig(weat_config_json_extracted)
            weat_configs[bias_category] = weat_config_object
        return weat_configs

    def run_weat(self, bias_categories):
        weat_configs = []
        for bias_category_name in bias_categories:
            weat_configs.append(self._weat_configs[bias_category_name])
        self._run_weat_internal(weat_configs)


    def run_weat_with_clusters(self, bias_categories_to_cluster):
        clusterer = EmbeddigsClusterer.create(self.model_handler.models[0], self._model_clusterer_config)
        weat_configs = []
        for bias_category_name in bias_categories_to_cluster:
            bias_category = self._weat_configs[bias_category_name]
            score_for_word_in_cluster = clusterer.calculate_score(bias_category)
            target_words_from_clusters = clusterer.get_target_words(score_for_word_in_cluster)
            for x_target_words, y_target_words in target_words_from_clusters:
                weat_config_for_cluster = bias_category.copy({
                    "x": x_target_words,
                    "y": y_target_words
                })
                weat_configs.append(weat_config_for_cluster)
        self._run_weat_internal(weat_configs, [clusterer.model])

    def _run_weat_internal(self, weat_configs, models=None):
        self._logger.clear_log(LOG_SUFFIX)
        self._logger.clear_log(RESULTS_SUFFIX) # delete old entries in the results-file
        self._logger.clear_log(FULL_RESULTS_SUFFIX)  # delete old entries in the results-file
        for weat_config in weat_configs:
            try:
                # try for each corpus
                full_test_results = self.bias_assessor.run_bias_test(weat_config,
                                                                     self._model_config.number_of_permutations,
                                                                     self.log_exception,
                                                                     models)
                evaluated_test_results = Evaluator.evaluate_mean(full_test_results)
                self._logger.test_result_dump(RESULTS_SUFFIX, evaluated_test_results, True)
                self._logger.test_results_dump(FULL_RESULTS_SUFFIX, full_test_results, True)
            except BiasAssessorException as e:
                message = str(e)
                self._logger.log(LOG_SUFFIX, message)
            except Exception as e:
                message = str(e)
                self._logger.log(LOG_SUFFIX, message)

    def log_exception(self, exception):
        message = str(exception)
        self._logger.log(LOG_SUFFIX, message)

    @property
    def config(self):
        return self._config

    @property
    def model_handler(self):
        return self._model_handler

    @property
    def bias_assessor(self):
        return self._bias_assessor
