class TestResult:

    def __init__(self, bias_category, p_value, cohens_d, number_of_permutations, total_time, absent_words, used_words):
        self._bias_category = bias_category
        self._p_value = p_value
        self._cohens_d = cohens_d
        self._number_of_permutations = number_of_permutations
        self._total_time = total_time
        self._absent_words = absent_words
        self._used_words = used_words

    @staticmethod
    def create(bias_category, p_value, cohens_d, number_of_permutations, total_time, absent_words, used_words):
        test_result = TestResult(bias_category, p_value, cohens_d, number_of_permutations, total_time, absent_words, used_words)
        return test_result

    @property
    def bias_category(self):
        return self._bias_category

    @property
    def p_value(self):
        return self._p_value

    @property
    def cohens_d(self):
        return self._cohens_d

    @property
    def number_of_permutations(self):
        return self._number_of_permutations

    @property
    def total_time(self):
        return self._total_time

    @property
    def absent_words(self):
        return self._absent_words

    @property
    def used_words(self):
        return self._used_words