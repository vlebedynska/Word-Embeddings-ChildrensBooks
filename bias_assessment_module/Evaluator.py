from preconditions import preconditions

from bias_assessment_module.TestResult import TestResult


class Evaluator():
    """
    A class that implements a helper function to calculate a mean score for multiple test results per bias category.
    """

    @staticmethod
    @preconditions(
        lambda category_test_results: len(category_test_results) > 0
    )
    def evaluate_mean(category_test_results):
        """
        calculates means of the p-value, Cohen's d, number of permuations and total time
        :param category_test_results: [[]]
        :return: new TestResult object
        """
        mean_result = TestResult.create("",0,0,0,0,[],[])
        mean_result.bias_category = category_test_results[0].bias_category
        for model_result in category_test_results:
            mean_result.p_value += model_result.p_value/len(category_test_results)
            mean_result.cohens_d += model_result.cohens_d/len(category_test_results)
            mean_result.number_of_permutations += model_result.number_of_permutations/len(category_test_results)
            mean_result.total_time += model_result.total_time/len(category_test_results)
        return mean_result
