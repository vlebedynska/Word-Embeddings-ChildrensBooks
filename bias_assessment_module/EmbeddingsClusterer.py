import numpy as np
from sklearn.cluster import MiniBatchKMeans

from bias_assessment_module.BiasAssessor import BiasAssessor
from bias_assessment_module.Utils import Utils


class EmbeddigsClusterer:
    """
    implements the unsupervised approach to finding new target word sets from the embedding's own vocabulary
     using the k-means++ algorithm for clustering word embeddings.
    """

    def __init__(self, cluster2word, model, config):
        self._model = model
        self._config = config
        self._cluster2word = cluster2word

    @staticmethod
    def create(model, config):
        """
        instantiates the EmbeddingsClusterer object by applying the MiniBatchKMeans
        algorithm to the model vectors and assigning each word from the model vocabulary to a cluster.
        :param model: word embeddings model
        :param config: EmbeddingsClustererConfig object
        :return: new EmbeddingsClusterer instance
        """
        # create an instance of MiniBatchKMeans
        mbk = MiniBatchKMeans(init=config.init, n_clusters=config.n_clusters, batch_size=config.batch_size,
                              max_no_improvement=config.max_no_improvement, verbose=config.verbose)
        # cluster the word embeddings of the model
        mbk.fit(model.wv.vectors)
        cluster2word = {}
        for word, cluster in zip(model.wv.vocab, mbk.labels_):
            cluster_record = cluster2word.get(cluster, None)
            if cluster_record is None:
                cluster_record = []
                cluster2word[cluster] = cluster_record
            cluster_record.append(word)
        return EmbeddigsClusterer(cluster2word, model, config)

    def calculate_score(self, weat_config):
        """
        calculates score for each word in a cluster.
        :param weat_config: WeatConfig object
        :return: map cluster => word scores grouped and mapped by cluster id
        """
        score_for_word_in_cluster = {}
        # filter out attribute words from the list A that are not in model vocabulary.
        a_attrs, _ = Utils.filter_list(self._model.wv.vocab, weat_config.a)
        # filter out attribute words from the list B that are not in model vocabulary.
        b_attrs, _ = Utils.filter_list(self._model.wv.vocab, weat_config.b)
        for cluster in sorted(self._cluster2word.keys()):
            cluster_words = self._cluster2word[cluster]
            word_score = []
            for word in cluster_words:
                # calculate the cosine mean difference between the vector of the word in the cluster
                # and vectors of the attribute words in list A and vectors of the attribute words in list B.
                score = BiasAssessor.cosine_means_difference(self._model.wv, word, a_attrs, b_attrs)
                word_score.append((word, score))
            score_for_word_in_cluster.update({cluster: word_score})
        return score_for_word_in_cluster

    def get_target_words(self, score_for_word_in_cluster):
        """
        creates target words lists X and Y.
        :param score_for_word_in_cluster: map cluster => word scores grouped and mapped by cluster id
        :return: list of tuples consisting of the target word list X and the target word list Y.
        """
        target_words = []
        for cluster, word_score in score_for_word_in_cluster.items():
            y_target_words = []
            y_target_scores = []
            x_target_words = []
            x_target_scores = []
            # Based on the scores for each word in a cluster,
            # add the word either to the target word list X or to the target word list Y.
            for word, score in word_score:
                if score < 0:
                    y_target_words.append(word)
                    y_target_scores.append(score * -1)  # convert score to a positive value
                else:
                    x_target_words.append(word)
                    x_target_scores.append(score)
            # select the words with the highest score to create the final X and Y target words sets
            x_target_words, y_target_words = self.choose_words(x_target_words, y_target_words, x_target_scores,
                                                               y_target_scores, self._config.cluster_words_count)
            target_words.append((x_target_words, y_target_words))
        return target_words

    def prepare_config_for_weat(self, bias_category):
        """
        is never used??
        :param bias_category: bias category name
        :return:
        """
        score_for_word_in_cluster = self.calculate_score(bias_category)
        target_words_from_clusters = self.get_target_words(score_for_word_in_cluster)
        for x_target_words, y_target_words in target_words_from_clusters:
            weat_config_for_cluster = bias_category.copy({
                "x": x_target_words,
                "y": y_target_words
            })
            return weat_config_for_cluster

    @staticmethod
    def choose_words(x_target_words, y_target_words, x_target_scores, y_target_scores, nwords=-1):
        """
        selects the best X and Y target words based on the word scores.
        :param x_target_words: list X of target words
        :param y_target_words: list Y of target words
        :param x_target_scores: list of scores for each word in the X target word list
        :param y_target_scores: list of scores for each word in the Y target word list
        :param nwords: maximum number of words in the single target word set
        :return: tuple of target word sets X and Y, each consisting of selected target words
        """
        n = np.min([len(x_target_words), len(y_target_words)] + ([nwords] if nwords >= 0 else []))
        x_target_scores = np.array(x_target_scores)
        y_target_scores = np.array(y_target_scores)
        x_topn_indices = x_target_scores.argsort()[::-1][:n]
        y_topn_indices = y_target_scores.argsort()[::-1][:n]
        sel_x_words = [x_target_words[i] for i in x_topn_indices]
        sel_y_words = [y_target_words[i] for i in y_topn_indices]
        return sel_x_words, sel_y_words

    @property
    def model(self):
        return self._model
