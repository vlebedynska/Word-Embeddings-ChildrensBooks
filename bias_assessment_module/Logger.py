class Logger():
    """
    A class that provides logging functionality used for dumping the test results to a file and on the console.
    """
    def __init__(self, model_id):
        self._model_id = model_id

    def clear_log(self, LOG_SUFFIX):
        """
        clears the contents of the file.
        :param LOG_SUFFIX: suffix of the file to be cleared
        :return: None
        """
        self.log(LOG_SUFFIX, "", False)

    def log(self, LOG_SUFFIX, message, append_to_file=True):
        """
        writes a log message to a file and to the console.
        :param LOG_SUFFIX: suffix of the file to which the message is to be written
        :param message: text of the message
        :param append_to_file: if True append to the existing file, otherwise create a new file
        :return: None
        """
        mode = "a" if append_to_file else "w"
        with open(self._model_id + LOG_SUFFIX, mode) as file:
            file.write(message)
        print(message)

    def test_result_dump(self, file_suffix, test_result, append_to_file=False):
        """
        writes single test result to a file.
        :param file_suffix: suffix of the file to which the result is to be written
        :param test_result: WEAT result
        :param append_to_file: if True append to the existing file, otherwise create a new file
        :return: None
        """
        self.test_results_dump(file_suffix, [test_result], append_to_file)

    def test_results_dump(self, file_suffix, test_results, append_to_file=False):
        """
        writes a list of test results to a file.
        :param file_suffix: suffix of the file to which the results are to be written
        :param test_results: list of WEAT results
        :param append_to_file: if True append to the existing file, otherwise create a new file
        :return: None
        """
        mode = "a" if append_to_file else "w"
        with open(self._model_id + file_suffix, mode) as file:
            for test_result in test_results:
                result_string = self._model_id + " {_bias_category}\t{_p_value}\t{_cohens_d:.4f}\t{_number_of_permutations}\t{_total_time:.4f}\t{_absent_words}\t{_used_words}\n".format(
                    **vars(test_result))
                file.write(result_string)
                print(result_string)