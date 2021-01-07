import math
import random
import time
from preconditions import preconditions

from pandas import np

from bias_assessment_module.BiasAssessorException import BiasAssessorException
from bias_assessment_module.TestResult import TestResult
from bias_assessment_module.Utils import Utils


class BiasAssessor:
    def __init__(self, models, config):
        self._models = models
        self._config = config

    @staticmethod
    def create(models, config):
        return BiasAssessor(models, config)

    def bias_test_for_clusters(self, attr_a, attr_b, target_words_from_clusters, bias_category):
        test_results = []
        for a_target_words, b_target_words in target_words_from_clusters:
            for model in self._models:
                if len(a_target_words) == 0 or len(b_target_words) == 0:
                    continue
                test_results.append(self.bias_test(
                    self._models[0],
                    attr_a,
                    attr_b,
                    a_target_words,
                    b_target_words,
                    bias_category
                ))
        return test_results

    def start_bias_test(self, bias_category):
        category_test_results = []
        for model in self._models:
            category_test_result = self.bias_test(
                model,
                self._config["lists"][bias_category]["attr"]["a"],
                self._config["lists"][bias_category]["attr"]["b"],
                self._config["lists"][bias_category]["target"]["x"],
                self._config["lists"][bias_category]["target"]["y"],
                bias_category
            )
            category_test_results.append(category_test_result)
        return category_test_results

    # TODO add subcategories for bias_categories if needed -> for this define "default" bias_subcategory as constant
    #  value and change config.json


    @preconditions(
        lambda attr_a: len(attr_a) > 0,
        lambda attr_b: len(attr_b) > 0,
        lambda target_x: len(target_x) > 0,
        lambda target_y: len(target_y) > 0
    )
    def bias_test(self, model, attr_a, attr_b, target_x, target_y, bias_category):
        start_time = time.time()
        wv = model.wv
        number_of_permutations = self._config["number_of_permutations"]
        a_attrs, a_filtered_out = Utils.filter_list(wv.vocab, attr_a)
        b_attrs, b_filtered_out = Utils.filter_list(wv.vocab, attr_b)
        x_targets, x_filtered_out = Utils.filter_list(wv.vocab, target_x)
        y_targets, y_filtered_out = Utils.filter_list(wv.vocab, target_y)
        if Utils.one_is_empty(a_attrs, b_attrs, x_targets, y_targets):
            args = {"a_attr": a_attrs, "b_attr": b_attrs, "x_targets": x_targets, "y_targets": y_targets}
            raise BiasAssessorException([attr_name for attr_name in args if len(args[attr_name]) == 0], bias_category)
        else:
            # balance the number of elements in the sets of two concepts
            a_attrs, b_attrs = Utils.balance_sets(a_attrs, b_attrs)
            x_targets, y_targets = Utils.balance_sets(x_targets, y_targets)

            p_value = BiasAssessor.weat_rand_test(wv, x_targets, y_targets, a_attrs, b_attrs, number_of_permutations)
            cohens_d = BiasAssessor.get_cohens_d(wv, x_targets, y_targets, a_attrs, b_attrs)
            used = [a_attrs, b_attrs, x_targets, y_targets]
            absent = [a_filtered_out, b_filtered_out, x_filtered_out, y_filtered_out]
            total_time = time.time() - start_time
            test_result = TestResult.create(bias_category, p_value, cohens_d, number_of_permutations, total_time, absent, used)
            return test_result



    @staticmethod
    def weat_rand_test(wv, x_targets, y_targets, a_attrs, b_attrs, iterations):
        u_words = x_targets + y_targets
        runs = np.min((iterations, math.factorial(len(u_words))))
        seen = set()

        original = BiasAssessor.test_statistic(wv, x_targets, y_targets, a_attrs, b_attrs)
        r = 0
        for i in range(runs):
            permutation = tuple(random.sample(u_words, len(u_words)))
            if i % 1000 == 0:
                print("permutation number " + str(i) + "/" + str(runs))
            if permutation not in seen:
                x_hat = permutation[0:len(x_targets)]
                y_hat = permutation[len(y_targets):]
                if BiasAssessor.test_statistic(wv, x_hat, y_hat, a_attrs, b_attrs) > original:
                    r += 1
                seen.add(permutation)
        p_value = r / runs
        return p_value

    @staticmethod
    def get_cohens_d(wv, x_targets, y_targets, a_attrs, b_attrs):
        if len(x_targets) == 0 or len(y_targets) == 0:
            return "NA"
        x_sum, y_sum = BiasAssessor.test_sums(wv, x_targets, y_targets, a_attrs, b_attrs)
        x_mean = x_sum / len(x_targets)
        y_mean = y_sum / len(y_targets)
        x_u_y = np.array([BiasAssessor.cosine_means_difference(wv, w, a_attrs, b_attrs) for w in x_targets + y_targets])
        stdev = x_u_y.std(ddof=1)
        return (x_mean - y_mean) / stdev

    @staticmethod
    def test_statistic(wv, x_targets, y_targets, a_attrs, b_attrs):
        b_sum, a_sum = BiasAssessor.test_sums(wv, x_targets, y_targets, a_attrs, b_attrs)
        return b_sum - a_sum

    @staticmethod
    def test_sums(wv, x_targets, y_targets, a_attrs, b_attrs):
        x_sum = 0.0
        y_sum = 0.0
        for t in x_targets:
            x_sum += BiasAssessor.cosine_means_difference(wv, t, a_attrs, b_attrs)
        for t in y_targets:
            y_sum += BiasAssessor.cosine_means_difference(wv, t, a_attrs, b_attrs)
        return x_sum, y_sum

    @staticmethod
    def cosine_means_difference(wv, word, a_attrs, b_attrs):
        a_mean = BiasAssessor.cosine_mean(wv, word, a_attrs)
        b_mean = BiasAssessor.cosine_mean(wv, word, b_attrs)
        result = a_mean - b_mean
        # print("current word: " + word + "\tmale_mean: " + str(male_mean) + "\tfemale_mean: " + str(female_mean) + "\t result: " + str(result))
        return result

    @staticmethod
    def cosine_mean(wv, word, attrs):
        return wv.cosine_similarities(wv[word], [wv[w] for w in attrs]).mean()