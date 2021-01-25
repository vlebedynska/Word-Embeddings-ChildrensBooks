import math
import random
import time
from preconditions import preconditions
import numpy as np

from bias_assessment_module.BiasAssessorException import BiasAssessorException
from bias_assessment_module.TestResult import TestResult
from bias_assessment_module.Utils import Utils


class BiasAssessor:
    """
    A class that implements the WEAT experimental protocol.
    """
    def __init__(self, models):
        self._models = models

    @staticmethod
    def create(models):
        """
        creates an instance of the BiasAssessor class.
        :param models: list of models
        :return: new BiasAssessor object
        """
        return BiasAssessor(models)

    def run_bias_test(self, bias_category, number_of_permutations, exception_logging_function, models=None):
        """
        executes the WEAT experimental protocol for each model and category passed.
        :param bias_category: name of the bias category
        :param number_of_permutations: number of permutations of the merged set of X and Y for the calculation of the one-sided p-value
        :param exception_logging_function: function for logging runtime exceptions to a file
        :param models: models used for the WEAT. Local instance models are used by default.
        :return: list of test results for all models
        """
        category_test_results = []
        if models is None:
            models = self._models
        for model in models:
            try:
                category_test_result = self._bias_test(
                    model,
                    bias_category.a,
                    bias_category.b,
                    bias_category.x,
                    bias_category.y,
                    bias_category.name,
                    number_of_permutations
                )
                category_test_results.append(category_test_result)
            except Exception as e:
                exception_logging_function(e)
        return category_test_results


    @preconditions(
        lambda attr_a: len(attr_a) > 0,
        lambda attr_b: len(attr_b) > 0,
        lambda target_x: len(target_x) > 0,
        lambda target_y: len(target_y) > 0
    )
    def _bias_test(self, model, attr_a, attr_b, target_x, target_y, bias_category, number_of_permutations):
        """
        executes the WEAT experimental protocol.
        :param model: word embeddings model on which the WEAT is to be performed
        :param attr_a: list A of attribute words
        :param attr_b: list B of attribute words
        :param target_x: list X of target words
        :param target_y: list Y of target words
        :param bias_category: name of the bias category
        :param number_of_permutations: number of permutations of the merged set of X and Y for the calculation of the one-sided p-value
        :raise BiasAssessorException
        :return: test result object
        """
        start_time = time.time()
        wv = model.wv
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
        """
        calculates the probability that the null hypothesis H0 is true. H0 proposes that there is no
         difference between X and Y in terms of their relative cosine similarity to A and B.
         The high p-value (p>0.005) suggests that the sets of target words X and Y are not biased against the concept.
        :param wv: model
        :param x_targets: list X of target words
        :param y_targets: list Y of target words
        :param a_attrs: list A of attribute words
        :param b_attrs: list B of attribute words
        :param iterations: number of permutations of the merged set of X and Y for the calculation of the one-sided p-value
        :return: p-value
        """
        u_words = x_targets + y_targets
        runs = np.min((iterations, math.factorial(len(u_words))))
        seen = set()
        # measure the original test statistic
        original = BiasAssessor.test_statistic(wv, x_targets, y_targets, a_attrs, b_attrs)
        r = 0
        # calculate the one-sided p-value
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
        # get proportion
        p_value = r / runs
        return p_value

    @staticmethod
    def get_cohens_d(wv, x_targets, y_targets, a_attrs, b_attrs):
        """
        calculates the effect size d based on the number of standard deviations that separate the two sets
        of target words in terms of their association with the attribute words.
        :param wv: model
        :param x_targets: list X of target words
        :param y_targets: list Y of target words
        :param a_attrs: list A of attribute words
        :param b_attrs: list B of attribute words
        :return: cohen's d
        """
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
        """
        measures the difference of the aggregated cosine similarities between the target word lists.
        :param wv: model
        :param x_targets: list X of target words
        :param y_targets: list Y of target words
        :param a_attrs: list A of attribute words
        :param b_attrs: list B of attribute words
        :return: test statistic
        """
        x_sum, y_sum = BiasAssessor.test_sums(wv, x_targets, y_targets, a_attrs, b_attrs)
        return x_sum - y_sum

    @staticmethod
    def test_sums(wv, x_targets, y_targets, a_attrs, b_attrs):
        """
        aggregates the cosine mean difference for each item in X and Y target word lists.
        :param wv: model
        :param x_targets: list X of target words
        :param y_targets: list Y of target words
        :param a_attrs: list A of attribute words
        :param b_attrs: list B of attribute words
        :return: a tuple of aggregated cosine mean differences (x_sum, y_sum)
        """
        x_sum = 0.0
        y_sum = 0.0
        for t in x_targets:
            x_sum += BiasAssessor.cosine_means_difference(wv, t, a_attrs, b_attrs)
        for t in y_targets:
            y_sum += BiasAssessor.cosine_means_difference(wv, t, a_attrs, b_attrs)
        return x_sum, y_sum

    @staticmethod
    def cosine_means_difference(wv, word, a_attrs, b_attrs):
        """
        measures the cosine difference between the embedding of the target word w ⃗ with the embeddings of the attribute words a ⃗∈A and b ⃗∈B.
        :param wv: model
        :param word: target word
        :param a_attrs: list A of attribute words
        :param b_attrs: list B of attribute words
        :return: cosine difference between the embedding of the target word w ⃗ with the embeddings of the attribute words a ⃗∈A and b ⃗∈B
        """
        a_mean = BiasAssessor.cosine_mean(wv, word, a_attrs)
        b_mean = BiasAssessor.cosine_mean(wv, word, b_attrs)
        result = a_mean - b_mean
        # print("current word: " + word + "\tmale_mean: " + str(male_mean) + "\tfemale_mean: " + str(female_mean) + "\t result: " + str(result))
        return result

    @staticmethod
    def cosine_mean(wv, word, attrs):
        """
        measures a mean of cosine similarities between the vector of a word and the vectors of all attribute words.
        :param wv: model
        :param word: word
        :param attrs: list of attribute words
        :return: mean cosine similarity between word and attrs
        """
        return wv.cosine_similarities(wv[word], [wv[w] for w in attrs]).mean()