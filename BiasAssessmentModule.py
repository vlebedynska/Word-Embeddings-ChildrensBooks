from BiasAssessor import BiasAssessor
from EmbeddingsClusterer import EmbeddigsClusterer
from ModelHandler import ModelHandler
import json


class BiasAssessmentModule():

    def run(self):

        with open("config.json", "r") as config_file:
            config = json.load(config_file)
        test_results = []
        model_handler = ModelHandler.create_and_load(config["model"])

        # delete old entries in the results-file
        BiasAssessmentModule.test_result_dump(model_handler.model_id, test_results)

        bias_assessor = BiasAssessor.create(model_handler.model, config["weat_lists"])
        test_results.append(bias_assessor.bias_test(
            config["weat_lists"]["lists"]["gender.b1"]["attr"]["female"],
            config["weat_lists"]["lists"]["gender.b1"]["attr"]["male"],
            config["weat_lists"]["lists"]["gender.b1"]["target"]["x"],
            config["weat_lists"]["lists"]["gender.b1"]["target"]["y"],
            "gender.b1"
        ))
        BiasAssessmentModule.test_result_dump(model_handler.model_id, test_results, True)
        print(format(model_handler.model.wv.most_similar(positive="girl", topn=10)))
        print(format(model_handler.model.wv.similarity('queen', 'weak')))


        clusterer = EmbeddigsClusterer.create(model_handler.model, config["clustering"])
        score_for_word_in_cluster = clusterer.calculate_score(config["weat_lists"]["lists"]["gender.b1"])
        target_words_from_clusters = clusterer.get_target_words(score_for_word_in_cluster)
        cluster_test_results = bias_assessor.bias_test_for_clusters(
            config["weat_lists"]["lists"]["gender.b1"]["attr"]["female"],
            config["weat_lists"]["lists"]["gender.b1"]["attr"]["male"],
            target_words_from_clusters,
            "gender.b1")
        BiasAssessmentModule.test_result_dump(model_handler.model_id, cluster_test_results, True)


    @staticmethod
    def test_result_dump(model_name, test_results, append_to_file=False):
        mode = "a" if append_to_file else "w"
        with open(model_name + "_results.txt", mode) as file:
            for test_result in test_results:
                BiasAssessmentModule.prettify_test_result(test_result)
                file.write("{_bias_category}\t{_p_value}\t{_cohens_d:.4f}\t{_number_of_permutations}\t{_total_time:.4f}\t{_absent_words}\t{_used_words}\n"
                           .format(**vars(test_result)))

    @staticmethod
    def prettify_test_result(test_result):
        print("Bias Category: {}\t p-value: {}\t cohen's d: {}\t absent words: {}"\
            .format(test_result.bias_category, test_result.p_value, test_result.cohens_d, test_result.absent_words))

if __name__ == '__main__':
    module = BiasAssessmentModule()
    module.run()
