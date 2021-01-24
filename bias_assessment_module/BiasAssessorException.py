class BiasAssessorException(Exception):
    """
    A class that defines an exception in the case of an error occurring during the WEAT execution.
    """
    def __init__(self, missing_attr_target_lists, bias_category):
        """
        creates an exception in case of error occurrence during the WEAT execution.
        :param missing_attr_target_lists: list of missing target and/or attribute words
        :param bias_category: name of the bias category
        """
        super(BiasAssessorException, self).__init__(', '.join(missing_attr_target_lists) + " list/s is/are empty in the bias category " + bias_category)
        self.bias_category = bias_category
        self.missing_attr_target_lists = missing_attr_target_lists
