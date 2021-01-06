import argparse
from bias_assessment_module.BiasAssessmentModule import BiasAssessmentModule
from bias_assessment_module.BiasAssessorException import BiasAssessorException
from bias_assessment_module.EmbeddingsClusterer import EmbeddigsClusterer
from bias_assessment_module.Evaluator import Evaluator


class WeatTester:

    def __init__(self, module, bias_categories):
        self._module = module
        self._bias_categories = bias_categories

    def run_weat_test(self):
        BiasAssessmentModule.test_results_dump(self._module.model_handler.model_id, "_results.txt", [])  # delete old entries in the results-file
        BiasAssessmentModule.test_results_dump(self._module._model_handler.model_id, "_results_full.txt", [])  # delete old entries in the results-file
        for bias_category in self._bias_categories:
            try:
                #try for each corpus
                full_test_results = self._module.bias_assessor.start_bias_test(bias_category)
                evaluated_test_results = Evaluator.evaluate_mean(full_test_results)
                BiasAssessmentModule.test_result_dump(self._module.model_handler.model_id, "_results.txt", evaluated_test_results, True)
                BiasAssessmentModule.test_results_dump(self._module.model_handler.model_id, "_results_full.txt", full_test_results, True)
            except BiasAssessorException as e:
                print(e)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--corpus', type=str, choices=['CLLIP_Corpus', 'ft'], default='w2v', help="Corpus name")
    return parser.parse_args()

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('-c', '--corpus', type=str, default='w2v', help="Corpus name")
    # parser.parse_args()

    print(parse_args())
    module = BiasAssessmentModule("config.json")
    bias_categories = [("gender.b1", True),
                       ("gender.b2", False),
                       ("gender.b3", False),
                       ("gender.b4", False),
                       ("gender.b5", False),
                       ("flowers_vs_insects", False),
                       ("animals", False),
                       ("dog_cat", False),
                       ("race", False),
                       ("gender_math", False),
                       ("age", True),
                       ("religion_with_names", False),
                       ("religion_christianity_islam", False),
                       ("religion_christianity_judaism", False),
                       ("religion_judaism_islam", False),
                       ("wolf_sheep", False),
                       ("fox_bird", False),
                       ("lion_mouse", False)
                       ]

    bias_categories_for_weat_test = []
    for bias_category, active in bias_categories:
        if active:
            bias_categories_for_weat_test.append(bias_category)


    weat_tester = WeatTester(module, bias_categories_for_weat_test)
    weat_tester.run_weat_test()


    clusterer = EmbeddigsClusterer.create(module.model_handler.models[0], module.config["clustering"])
    score_for_word_in_cluster = clusterer.calculate_score(module.config["weat_lists"]["lists"]["gender.b1"])
    target_words_from_clusters = clusterer.get_target_words(score_for_word_in_cluster)
    cluster_test_results = module.bias_assessor.bias_test_for_clusters(
        module.config["weat_lists"]["lists"]["gender.b1"]["attr"]["a"],
        module.config["weat_lists"]["lists"]["gender.b1"]["attr"]["b"],
        target_words_from_clusters,
        "gender.b1")
    BiasAssessmentModule.test_result_dump(module.model_handler.model_id, "_results_clusters.txt", cluster_test_results, True)

