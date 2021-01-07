from pandas import np
from sklearn.cluster import MiniBatchKMeans

from bias_assessment_module.BiasAssessor import BiasAssessor

class EmbeddigsClusterer:
    def __init__(self, cluster2word, model, config):
        self._model = model
        self._config = config
        self._cluster2word = cluster2word

    @staticmethod
    def create(model, config):

        mbk = MiniBatchKMeans(init=config["init"], n_clusters=config["n_clusters"], batch_size=config["batch_size"],
                              max_no_improvement=config["max_no_improvement"], verbose=["verbose"])
        mbk.fit(model.wv.vectors)
        cluster2word = {}
        for word, cluster in zip(model.wv.vocab, mbk.labels_):
            cluster_record = cluster2word.get(cluster, None)
            if cluster_record is None:
                cluster_record = []
                cluster2word[cluster] = cluster_record
            cluster_record.append(word)
        return EmbeddigsClusterer(cluster2word, model, config)

    def calculate_score(self, config):
        score_for_word_in_cluster = {}
        for cluster in sorted(self._cluster2word.keys()):
            cluster_words = self._cluster2word[cluster]
            word_score = []
            for word in cluster_words:
                score = BiasAssessor.cosine_means_difference(self._model.wv, word, config["attr"]["a"], config["attr"]["b"])
                word_score.append((word, score))
            score_for_word_in_cluster.update({cluster: word_score})
        return score_for_word_in_cluster

    def get_target_words(self, score_for_word_in_cluster):
        target_words = []
        for cluster, word_score in score_for_word_in_cluster.items():
            y_target_words = []
            y_target_scores = []
            x_target_words = []
            x_target_scores = []
            for word, score in word_score:
                if score < 0:
                    y_target_words.append(word)
                    y_target_scores.append(score * -1)  # convert score to a positive value
                else:
                    x_target_words.append(word)
                    x_target_scores.append(score)
            x_target_words, y_target_words = self.choose_words(x_target_words, y_target_words, x_target_scores, y_target_scores, self._config["cluster_words_count"])
            target_words.append((x_target_words, y_target_words))
        return target_words


    @staticmethod
    def choose_words(x_target_words, y_target_words, x_target_scores, y_target_scores, nwords=-1):
        n = np.min([len(x_target_words), len(y_target_words)] + ([nwords] if nwords >= 0 else []))
        x_target_scores = np.array(x_target_scores)
        y_target_scores = np.array(y_target_scores)
        x_topn_indices = x_target_scores.argsort()[::-1][:n]
        y_topn_indices = y_target_scores.argsort()[::-1][:n]
        sel_x_words = [x_target_words[i] for i in x_topn_indices]
        sel_y_words = [y_target_words[i] for i in y_topn_indices]
        return sel_x_words, sel_y_words