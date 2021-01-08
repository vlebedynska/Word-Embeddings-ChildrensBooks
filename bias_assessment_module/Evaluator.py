from preconditions import preconditions

from bias_assessment_module.TestResult import TestResult


class Evaluator():
    """implements function for creating the mean result from the WEAT results for each bias category

    Methods
    _______
    evaluate_mean(category_test_results)
        calculates means of the p-value,cohens d, number of permutations and total time
    """

    @staticmethod
    @preconditions(
        lambda category_test_results: len(category_test_results) > 0
    )
    def evaluate_mean(category_test_results):
        """
        calculates means of the p-value, cohens_d, number of permuations and total time

        :param category_test_results: [[]]
        :return: TestResult Object in an array
        """
        mean_result = TestResult.create("",0,0,0,0,[],[])
        mean_result.bias_category = category_test_results[0].bias_category
        for model_result in category_test_results:
            mean_result.p_value += model_result.p_value/len(category_test_results)
            mean_result.cohens_d += model_result.cohens_d/len(category_test_results)
            mean_result.number_of_permutations += model_result.number_of_permutations/len(category_test_results)
            mean_result.total_time += model_result.total_time/len(category_test_results)
        return mean_result
