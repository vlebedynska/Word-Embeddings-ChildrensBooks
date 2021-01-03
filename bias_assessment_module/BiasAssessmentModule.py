from bias_assessment_module.BiasAssessor import BiasAssessor
from bias_assessment_module.EmbeddingsClusterer import EmbeddigsClusterer
from bias_assessment_module.ModelHandler import ModelHandler
import json

from bias_assessment_module.TestResult import TestResult


class BiasAssessmentModule():
    def __init__(self, config):
        self._config = config
        with open(config, "r") as config_file:
            config = json.load(config_file)
        self._model_handler = ModelHandler.create_and_load(config["model"])
        self._bias_assessor = BiasAssessor.create(self._model_handler.models, config["weat_lists"])


    @property
    def model_handler(self):
        return self._model_handler

    @property
    def bias_assessor(self):
        return self._bias_assessor


    def run(self, config):
        pass

        # print(format(model_handler.model.wv.most_similar(positive="cat", topn=10)))
        # print(format(model_handler.model.wv.most_similar(positive="dog", topn=10)))
        # print(format(model_handler.model.wv.similarity('queen', 'king')))


        # clusterer = EmbeddigsClusterer.create(model_handler.model, config["clustering"])
        # score_for_word_in_cluster = clusterer.calculate_score(config["weat_lists"]["lists"]["gender.b1"])
        # target_words_from_clusters = clusterer.get_target_words(score_for_word_in_cluster)
        # cluster_test_results = bias_assessor.bias_test_for_clusters(
        #     config["weat_lists"]["lists"]["gender.b1"]["attr"]["a"],
        #     config["weat_lists"]["lists"]["gender.b1"]["attr"]["b"],
        #     target_words_from_clusters,
        #     "gender.b1")
        # BiasAssessmentModule.test_result_dump(model_handler.model_id, cluster_test_results, True)



    @staticmethod
    def test_result_dump(model_name, file_suffix, test_result, append_to_file=False):
        BiasAssessmentModule.test_results_dump(model_name, file_suffix, [test_result], append_to_file)


    @staticmethod
    def test_results_dump(model_name, file_suffix, test_results, append_to_file=False):
        mode = "a" if append_to_file else "w"
        with open(model_name + file_suffix, mode) as file:
            for model in test_results:
                BiasAssessmentModule.prettify_test_result(model_name, model)
                file.write(model_name+
                    " {_bias_category}\t{_p_value}\t{_cohens_d:.4f}\t{_number_of_permutations}\t{_total_time:.4f}\t{_absent_words}\t{_used_words}\n"
                        .format(**vars(model)))

    @staticmethod
    def prettify_test_result(model_name, test_result):
        print(model_name + " Bias Category: {_bias_category}\t p-value: {_p_value}\t cohen's d: {_cohens_d}\t absent words: {_absent_words}"\
            .format(**vars(test_result)))
