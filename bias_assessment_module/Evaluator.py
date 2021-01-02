from bias_assessment_module.TestResult import TestResult


class Evaluator():

    @staticmethod
    def evaluate_mean(full_test_results):
        test_results = []
        p_value = 0
        cohens_d = 0
        number_of_permutations = 0
        total_time = 0
        count = 0
        bias_category = ""
        absent_words = []
        used_words = []
        for result in full_test_results:
            for model in result:
                p_value = p_value + model.p_value
                cohens_d = cohens_d + model.cohens_d
                number_of_permutations = number_of_permutations + model.number_of_permutations
                total_time = total_time + model.total_time
                count = count + 1
                if not bias_category:
                    bias_category = model.bias_category
                if model.absent_words not in absent_words:
                    absent_words.append(model.absent_words)
                if model.used_words not in used_words:
                    used_words.append(model.used_words)
        test_results.append(Evaluator.create_normalized_test_result(
                bias_category, p_value, cohens_d, number_of_permutations, total_time, used_words, absent_words, count
            )
)
        return test_results


    @staticmethod
    def create_normalized_test_result(bias_category, p_value, cohens_d, number_of_permutations, total_time, used_words, absent_words, count):
        total_p_value, total_cohens_d, total_number_of_permutations, total_total_time = Evaluator.calculate_mean_test_result(p_value, cohens_d, number_of_permutations, total_time, count)
        return TestResult.create(bias_category, total_p_value, total_cohens_d, total_number_of_permutations, total_total_time, used_words, absent_words)


    @staticmethod
    def merge_words(absent_words):
        total_words = 0
        return total_words

    @staticmethod
    def calculate_mean_test_result(p_value, cohens_d, number_of_permutations, total_time, count):
        total_p_value = Evaluator.calculate_mean(p_value, count)
        total_cohens_d = Evaluator.calculate_mean(cohens_d, count)
        total_number_of_permutations = Evaluator.calculate_mean(number_of_permutations, count)
        total_total_time = Evaluator.calculate_mean(total_time, count)
        return total_p_value, total_cohens_d, total_number_of_permutations, total_total_time

    @staticmethod
    def calculate_mean(number, counter):
        return number/counter