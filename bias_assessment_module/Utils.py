import random


class Utils:
    """
    A class that implements various help functions for the WEAT
    experimental protocol, for instance for balancing the sets of target and attribute words.
    """
    @staticmethod
    def filter_list(list, filter):
        """
        splits a list in two lists (found / not found) by a given filter.
        :param list: list to be filtered
        :param filter: list of items that are used as a filter
        :return: tuple of two lists found_items and not_found_items
        """
        found_items = []
        not_found_items = []
        for item in filter:
            if item in list:
                found_items.append(item)
            else:
                not_found_items.append(item)
        return found_items, not_found_items


    @staticmethod
    def one_is_empty(a_attrs, b_attrs, x_targets, y_targets):
        """
        checks if one of the passed lists is empty.
        :param a_attrs: list A of list A of attribute words
        :param b_attrs: list B of attribute words
        :param x_targets: list X of target words
        :param y_targets: list Y of target words
        :return: False if the list is not empty, otherwise True.
        """
        list_of_length = []
        list_of_length.append(len(a_attrs))
        list_of_length.append(len(b_attrs))
        list_of_length.append(len(x_targets))
        list_of_length.append(len(y_targets))
        for item in list_of_length:
            if item != 0:
                continue
            else:
                return True
        return False

    @staticmethod
    def balance_sets(first_set, second_set):
        """
        creates two lists of equal size if the length the lists is not equal.
        :param first_set: first list
        :param second_set: second list
        :return: always returns a copy of the two lists
        """
        # number = (a_attrs - b_attrs) if len(a_attrs) >= len(b_attrs) else (b_attrs-a_attrs)
        if len(first_set) != len(second_set):
            set_size = min(len(first_set), len(second_set))
            first_set_copy = random.sample(first_set, k=set_size)
            second_set_copy = random.sample(second_set, k=set_size)
        else:
            first_set_copy = first_set.copy()
            second_set_copy = second_set.copy()
        return first_set_copy, second_set_copy

