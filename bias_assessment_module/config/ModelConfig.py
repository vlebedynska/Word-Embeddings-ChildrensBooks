class ModelConfig():
    """
    A class that holds configuration data for creating, saving and loading the word embedding model.
    """
    def __init__(self, config):
        self._config = config

    @property
    def corpus_path(self):
        return self._config["corpus_path"]

    @property
    def model_path(self):
        return self._config["model_path"]

    @property
    def corpus_name(self):
        return self._config["corpus_name"]

    @property
    def amount_of_corpora(self):
        return self._config["amount_of_corpora"]

    @property
    def size(self):
        return self._config["size"]

    @property
    def window(self):
        return self._config["window"]

    @property
    def sg(self):
        return self._config["sg"]

    @property
    def epochs(self):
        return self._config["epochs"]

    @property
    def number_of_permutations(self):
        return self._config["number_of_permutations"]