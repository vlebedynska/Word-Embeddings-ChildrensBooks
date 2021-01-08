class Logger():
    def __init__(self, model_id):
        self._model_id = model_id

    def clear_log(self, LOG_SUFFIX):
        self.log(LOG_SUFFIX, "", False)


    def log(self, LOG_SUFFIX, message, append_to_file=True):
        mode = "a" if append_to_file else "w"
        with open(self._model_id + LOG_SUFFIX, mode) as file:
            file.write(message)
        print(message)

    def test_result_dump(self, file_suffix, test_result, append_to_file=False):
        self.test_results_dump(file_suffix, [test_result], append_to_file)

    def test_results_dump(self, file_suffix, test_results, append_to_file=False):
        mode = "a" if append_to_file else "w"
        with open(self._model_id + file_suffix, mode) as file:
            for test_result in test_results:
                result_string = self._model_id + " {_bias_category}\t{_p_value}\t{_cohens_d:.4f}\t{_number_of_permutations}\t{_total_time:.4f}\t{_absent_words}\t{_used_words}\n".format(
                    **vars(test_result))
                file.write(result_string)
                print(result_string)