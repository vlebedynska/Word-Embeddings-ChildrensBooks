from BiasAssessor import BiasAssessor
from ModelHandler import ModelHandler
import json


class BiasAssessmentModule():

    def run(self):
        with open("config.json", "r") as config_file:
            config = json.load(config_file)
        test_results = []
        model_handler = ModelHandler.create_and_load(config["model"])
        bias_assessor = BiasAssessor.create(model_handler.model, config["weat_lists"])
        test_results.append(bias_assessor.bias_test("gender.b1"))
        BiasAssessmentModule.test_result_dump(model_handler.model_id, test_results)
        print(format(model_handler.model.wv.most_similar(positive="girl", topn=10)))
        print(format(model_handler.model.wv.similarity('queen', 'weak')))

    @staticmethod
    def test_result_dump(model_name, test_results):
        with open(model_name + "_results.txt", "w") as file:
            for test_result in test_results:
                BiasAssessmentModule.prettify_test_result(test_result)
                file.write("{}\t{}\t{:.4f}\t{}\t{:.4f}\t{}\t{}\n"
                           .format(test_result.bias_category, test_result.p_value, test_result.cohens_d,
                                   test_result.number_of_permutations, test_result.total_time, test_result.absent_words,
                                   test_result.used_words))

    @staticmethod
    def prettify_test_result(test_result):
        print("Bias Category: {}\t p-value: {}\t cohen's d: {}\t absent words: {}"\
            .format(test_result.bias_category, test_result.p_value, test_result.cohens_d, test_result.absent_words))

if __name__ == '__main__':
    module = BiasAssessmentModule()
    module.run()
