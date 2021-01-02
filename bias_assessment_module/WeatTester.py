from bias_assessment_module.BiasAssessmentModule import BiasAssessmentModule
from bias_assessment_module.BiasAssessorException import BiasAssessorException
from bias_assessment_module.Evaluator import Evaluator


class WeatTester:

    def __init__(self, module, bias_categories):
        self._module = module
        self._bias_categories = bias_categories

    def run_weat_test(self):
        for bias_category in self._bias_categories:
            try:
                #try for each corpus
                full_test_results = self._module.bias_assessor.start_bias_test(bias_category)
                evaluated_test_results = Evaluator.evaluate_mean(full_test_results)
                BiasAssessmentModule.test_result_dump(self._module.model_handler.model_id, evaluated_test_results, True)
            except BiasAssessorException as e:
                print(e)


if __name__ == '__main__':
    module = BiasAssessmentModule("config.json")
    bias_categories = [("gender.b1", False),
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
    for bias_categorie, active in bias_categories:
        if active:
            bias_categories_for_weat_test.append(bias_categorie)


    weat_tester = WeatTester(module, bias_categories_for_weat_test)
    weat_tester.run_weat_test()
