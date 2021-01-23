class TestResult:
    """
    A class that stores the WEAT result including the p-value and Cohen’s d.
    """

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
        """
        creates an instance of TestResult.
        :param bias_category: name of the bias category
        :param p_value: p-value
        :param cohens_d: Cohen' d
        :param number_of_permutations: number of permutations of the merged set of X and Y for the calculation of the one-sided p-value
        :param total_time: time needed for the WEAT
        :param absent_words: list of absent target and attribute words
        :param used_words: list of used target and attribute words
        :return: new TestResult object
        """
        test_result = TestResult(bias_category, p_value, cohens_d, number_of_permutations, total_time, absent_words, used_words)
        return test_result

    @property
    def bias_category(self):
        return self._bias_category

    @bias_category.setter
    def bias_category(self, bias_category):
        self._bias_category = bias_category

    @property
    def p_value(self):
        return self._p_value

    @p_value.setter
    def p_value(self, p_value):
        self._p_value = p_value

    @property
    def cohens_d(self):
        return self._cohens_d

    @cohens_d.setter
    def cohens_d(self, cohens_d):
        self._cohens_d = cohens_d

    @property
    def number_of_permutations(self):
        return self._number_of_permutations

    @number_of_permutations.setter
    def number_of_permutations(self, number_of_permutations):
        self._number_of_permutations = number_of_permutations

    @property
    def total_time(self):
        return self._total_time

    @total_time.setter
    def total_time(self, total_time):
        self._total_time = total_time

    @property
    def absent_words(self):
        return self._absent_words

    @property
    def used_words(self):
        return self._used_words