class EmbeddingsClustererConfig():
    """
    A class that holds configuration data for executing the k-means++ algorithm for clustering the vectors of the word embedding model.
    """
    def __init__(self, config):
        self._config = config

    @property
    def init(self):
        return self._config["init"]

    @property
    def n_clusters(self):
        return self._config["n_clusters"]

    @property
    def batch_size(self):
        return self._config["batch_size"]

    @property
    def max_no_improvement(self):
        return self._config["max_no_improvement"]

    @property
    def verbose(self):
        return self._config["verbose"]

    @property
    def cluster_words_count(self):
        return self._config["cluster_words_count"]
