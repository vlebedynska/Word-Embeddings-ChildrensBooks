from pandas import np
from sklearn.cluster import MiniBatchKMeans

from BiasAssessor import BiasAssessor


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
        score_for_word_in_cluster = []
        for cluster in sorted(self._cluster2word.keys()):
            cluster_words = self._cluster2word[cluster]
            for word in cluster_words:
                score = BiasAssessor.cosine_means_difference(self._model.wv, word, config["attr"]["male"], config["attr"]["female"])
                score_for_word_in_cluster.append((cluster, word, score))
        return score_for_word_in_cluster

    def get_m_f_word(self, score_for_word_in_cluster):
        last_cluster = None
        f_words = []
        f_scores = []
        m_words = []
        m_scores = []
        clustering_m_w_words = []
        for cluster, word, score in score_for_word_in_cluster:
            if score < 0:
                f_words.append(word)
                f_scores.append(score * -1)  # we convert it to a positive value
            else:
                m_words.append(word)
                m_scores.append(score)
            if last_cluster != cluster and last_cluster is not None:
                m_words, f_words = self.choose_words(m_words, f_words, m_scores, f_scores, -1)
                clustering_m_w_words.append((m_words, f_words))
                f_words = []
                m_words = []
                f_scores = []
                m_scores = []
            last_cluster = cluster
        m_words, f_words = self.choose_words(m_words, f_words, m_scores, f_scores, -1)
        clustering_m_w_words.append((m_words, f_words))
        return clustering_m_w_words


    @staticmethod
    def choose_words(m_words, f_words, m_scores, f_scores, nwords=-1):
        n = np.min([len(m_words), len(f_words)] + ([nwords] if nwords >= 0 else []))
        m_scores = np.array(m_scores)
        f_scores = np.array(f_scores)
        m_topn_indices = m_scores.argsort()[::-1][:n]
        f_topn_indices = f_scores.argsort()[::-1][:n]
        sel_m_words = [m_words[i] for i in m_topn_indices]
        sel_f_words = [f_words[i] for i in f_topn_indices]
        return sel_m_words, sel_f_words



    # def temp(self):
        # cluster2word = load_clusters("clustering.txt", model.wv.vocab)
        # with open("clustering_sorted", 'w') as fout:
        #     for cluster in sorted(cluster2word.keys()):
        #         cluster_words = cluster2word[cluster]
        #         cluster_desc = get_cluster_description(model.wv, cluster_words)
        #         for word in cluster_words:
        #             score = cosine_means_difference(model.wv, word, m, f)
        #             gender = 'F' if score < 0 else 'M'
        #             fout.write("{}\t{}\t{}\t{}\t{}\n".format(cluster, cluster_desc, word, gender, score))
        # print("Done!")
        #
        # last_cluster = None
        # f_words = []
        # f_scores = []
        # m_words = []
        # m_scores = []
        # genders = {'F': 0, 'M': 0}
        # with open("clustering_sorted", 'r') as fga, open("clusering_end", 'w') as fout:
        #     for lgassoc in fga:
        #         fields = lgassoc.strip().split('\t')
        #         cluster = int(fields[0])
        #         cluster_desc = fields[1]
        #         word = fields[2]
        #         gender_score = float(fields[4])
        #         if gender_score < 0:
        #             f_words.append(word)
        #             f_scores.append(gender_score * -1)  # we convert it to a positive value
        #             genders['F'] += 1
        #         else:
        #             m_words.append(word)
        #             m_scores.append(gender_score)
        #             genders['M'] += 1
        #         if last_cluster != cluster and last_cluster is not None:
        #             m_words, f_words = choose_words(m_words, f_words, m_scores, f_scores, -1)
        #             p_value = weat_rand_test(model.wv, m_words, f_words, m, f, 1000)
        #             cohens_d = get_cohens_d(model.wv, m_words, f_words, m, f)
        #             print_line(fout, cluster, maj_gender(genders), cluster_desc, m_words, f_words, p_value, cohens_d)
        #             f_words = []
        #             m_words = []
        #             f_scores = []
        #             m_scores = []
        #             genders = {'F': 0, 'M': 0}
        #         last_cluster = cluster
        #     m_words, f_words = choose_words(m_words, f_words, m_scores, f_scores, -1)
        #     p_value = weat_rand_test(model.wv, m_words, f_words, m, f, 1000) if not False else "?"
        #     cohens_d = get_cohens_d(model.wv, m_words, f_words, m, f)
        #     print_line(fout, cluster, maj_gender(genders), cluster_desc, m_words, f_words, p_value, cohens_d)
        # print("Done! 3")