from bias_assessment_module.BiasAssessmentModule import BiasAssessmentModule
from bias_assessment_module.BiasAssessorException import BiasAssessorException


class WeatTester:
    def __init__(self, module, bias_categories):
        self._module = module
        self._bias_categories = bias_categories

    def run_weat_test(self):
        for bias_category in self._bias_categories:
            try:
                test_results = self._module.bias_assessor.start_bias_test(bias_category)
                BiasAssessmentModule.test_result_dump(self._module.model_handler.model_id, test_results, True)
            except BiasAssessorException as e:
                print(e)


if __name__ == '__main__':
    module = BiasAssessmentModule("config.json")
    bias_categories = [("gender.b1", True),
                       ("gender.b2", True),
                       ("gender.b3", True),
                       ("gender.b4", True),
                       ("gender.b5", True),
                       ("flowers_vs_insects", False),
                       ("animals", False),
                       ("dog_cat", False),
                       ("race", False),
                       ("gender_math", False),
                       ("religion_with_names", False),
                       ("religion_christianity_islam", False),
                       ("religion_christianity_judaism", False),
                       ("religion_judaism_islam", False),
                       ("age", False)
                       ]

    bias_categories_for_weat_test = []
    for bias_categorie, active in bias_categories:
        if active:
            bias_categories_for_weat_test.append(bias_categorie)


    weat_tester = WeatTester(module, bias_categories_for_weat_test)
    weat_tester.run_weat_test()
