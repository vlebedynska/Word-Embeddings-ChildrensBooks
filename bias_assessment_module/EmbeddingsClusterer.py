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
                score = BiasAssessor.cosine_means_difference(self._model.wv, word, config["attr"]["male"], config["attr"]["female"])
                word_score.append((word, score))
            score_for_word_in_cluster.update({cluster: word_score})
        return score_for_word_in_cluster

    def get_target_words(self, score_for_word_in_cluster):
        target_words = []
        for cluster, word_score in score_for_word_in_cluster.items():
            a_target_words = []
            a_target_scores = []
            b_target_words = []
            b_target_scores = []
            for word, score in word_score:
                if score < 0:
                    a_target_words.append(word)
                    a_target_scores.append(score * -1)  # convert score to a positive value
                else:
                    b_target_words.append(word)
                    b_target_scores.append(score)
            b_target_words, a_target_words = self.choose_words(b_target_words, a_target_words, b_target_scores, a_target_scores, self._config["cluster_words_count"])
            target_words.append((b_target_words, a_target_words))
        return target_words


    @staticmethod
    def choose_words(b_target_words, a_target_words, b_target_scores, a_target_scores, nwords=-1):
        n = np.min([len(b_target_words), len(a_target_words)] + ([nwords] if nwords >= 0 else []))
        b_target_scores = np.array(b_target_scores)
        a_target_scores = np.array(a_target_scores)
        m_topn_indices = b_target_scores.argsort()[::-1][:n]
        f_topn_indices = a_target_scores.argsort()[::-1][:n]
        sel_m_words = [b_target_words[i] for i in m_topn_indices]
        sel_f_words = [a_target_words[i] for i in f_topn_indices]
        return sel_m_words, sel_f_words
