import math
import random
import time

from pandas import np

from TestResult import TestResult


class BiasAssessor:
    def __init__(self, model, config):
        self._model = model
        self._config = config

    @staticmethod
    def create(model, config):
        return BiasAssessor(model, config)

    # TODO add subcategories for bias_categories if needed -> for this define "default" bias_subcategory as constant
    #  value and change config.json

    def bias_test(self, bias_category):
        start_time = time.time()
        wv = self._model.wv
        category_data = self._config["lists"][bias_category]
        number_of_permutations = self._config["number_of_permutations"]
        f = category_data["attr"]["female"]
        m = category_data["attr"]["male"]
        x = category_data["target"]["x"]
        y = category_data["target"]["y"]
        f_attrs, f_filtered = BiasAssessor.filter_words(wv, f)
        m_attrs, m_filtered = BiasAssessor.filter_words(wv, m)
        x_targets, x_filtered = BiasAssessor.filter_words(wv, x)
        y_targets, y_filtered = BiasAssessor.filter_words(wv, y)
        print("f_attrs: " + str(f_attrs))
        print("m_attrs: " + str(m_attrs))
        print("x_targets: " + str(x_targets))
        print("y_targets: " + str(y_targets))
        p_value = BiasAssessor.weat_rand_test(wv, x_targets, y_targets, m_attrs, f_attrs, number_of_permutations)
        cohens_d = BiasAssessor.get_cohens_d(wv, x_targets, y_targets, m_attrs, f_attrs)
        print("{}\t{}\t{}\n".format("strength vs. weakness", p_value, cohens_d))
        used = [f_attrs, m_attrs, x_targets, y_targets]
        absent = [f_filtered, m_filtered, x_filtered, y_filtered]
        end_time = time.time()
        total_time = end_time - start_time
        test_result = TestResult.create(bias_category, p_value, cohens_d, number_of_permutations, total_time, absent, used)
        return test_result


    @staticmethod
    def filter_words(wv, words):
        final_words = []
        filtered_words = []
        for word in words:
            if word in wv.vocab:
                final_words.append(word)
            else:
                filtered_words.append(word)
        return final_words, filtered_words

    @staticmethod
    def weat_rand_test(wv, m_words, f_words, m_attrs, f_attrs, iterations):
        u_words = m_words + f_words
        runs = np.min((iterations, math.factorial(len(u_words))))
        seen = set()

        original = BiasAssessor.test_statistic(wv, m_words, f_words, m_attrs, f_attrs)
        r = 0
        for _ in range(runs):
            permutation = tuple(random.sample(u_words, len(u_words)))
            if permutation not in seen:
                m_hat = permutation[0:len(m_words)]
                f_hat = permutation[len(f_words):]
                if BiasAssessor.test_statistic(wv, m_hat, f_hat, m_attrs, f_attrs) > original:
                    r += 1
                seen.add(permutation)
        p_value = r / runs
        return p_value

    @staticmethod
    def get_cohens_d(wv, m_targets, f_targets, m_attrs, f_attrs):
        if len(m_targets) == 0 or len(f_targets) == 0:
            return "NA"
        m_sum, f_sum = BiasAssessor.test_sums(wv, m_targets, f_targets, m_attrs, f_attrs)
        m_mean = m_sum / len(m_targets)
        f_mean = f_sum / len(f_targets)
        m_u_f = np.array([BiasAssessor.cosine_means_difference(wv, w, m_attrs, f_attrs) for w in m_targets + f_targets])
        stdev = m_u_f.std(ddof=1)
        return (m_mean - f_mean) / stdev

    @staticmethod
    def test_statistic(wv, m_targets, f_targets, m_attrs, f_attrs):
        m_sum, f_sum = BiasAssessor.test_sums(wv, m_targets, f_targets, m_attrs, f_attrs)
        return m_sum - f_sum

    @staticmethod
    def test_sums(wv, m_targets, f_targets, m_attrs, f_attrs):
        m_sum = 0.0
        f_sum = 0.0
        for t in m_targets:
            m_sum += BiasAssessor.cosine_means_difference(wv, t, m_attrs, f_attrs)
        for t in f_targets:
            f_sum += BiasAssessor.cosine_means_difference(wv, t, m_attrs, f_attrs)
        return m_sum, f_sum

    @staticmethod
    def cosine_means_difference(wv, word, male_attrs, female_attrs):
        male_mean = BiasAssessor.cosine_mean(wv, word, male_attrs)
        female_mean = BiasAssessor.cosine_mean(wv, word, female_attrs)
        result = male_mean - female_mean
        # print("current word: " + word + "\tmale_mean: " + str(male_mean) + "\tfemale_mean: " + str(female_mean) + "\t result: " + str(result))
        return result

    @staticmethod
    def cosine_mean(wv, word, attrs):
        return wv.cosine_similarities(wv[word], [wv[w] for w in attrs]).mean()