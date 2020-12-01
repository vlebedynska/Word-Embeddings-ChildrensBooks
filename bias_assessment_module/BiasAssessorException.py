class BiasAssessorException(Exception):
    def __init__(self, missing_attr_target_lists, bias_category):
        super(BiasAssessorException, self).__init__(', '.join(missing_attr_target_lists) + " list/s is/are empty in the bias category " + bias_category)
        self.bias_category = bias_category
        self.missing_attr_target_lists = missing_attr_target_lists
