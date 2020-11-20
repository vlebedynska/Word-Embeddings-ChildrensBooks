class Utils:

    @staticmethod
    def filter_list(list, filter):
        found_items = []
        not_found_items = []
        for item in filter:
            if item in list:
                found_items.append(item)
            else:
                not_found_items.append(item)
        return found_items, not_found_items