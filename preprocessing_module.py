import math
import os
import random

import gensim
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import MiniBatchKMeans

from ModelHandler import ModelHandler


def load(model_name, corpus_name, modeltype, force_training=False):
    if modeltype == "w2v":
        if not os.path.exists(model_name) or force_training:
            return save(model_name, corpus_name)
        else:
            return Word2Vec.load(model_name)
    elif modeltype == "ft":
        return gensim.models.fasttext.load_facebook_vectors(model_name)
    else:
        raise ValueError("Unsupported Word Embedding type '{}'".format(modeltype))

def save(model_name, corpus_name):
    text_file = open(corpus_name, "r")
    documents = text_file.read().split('_BOOK_TITLE_')
    del documents[0]
    output_text = []
    for document in documents:
        output_text.append(gensim.utils.simple_preprocess(document))
    print(output_text[0])
    model = gensim.models.Word2Vec(output_text, size=150, window=10, min_count=2, workers=4)
    model.train(output_text, total_examples=len(output_text), epochs=10)
    model.save(model_name)
    return model


def gender_bias_test(x, y, m, f, wv):
    f_attrs = filter_words(wv, f)
    m_attrs = filter_words(wv, m)
    x_targets = filter_words(wv, x)
    y_targets = filter_words(wv, y)
    print("f_attrs: " + str(f_attrs))
    print("m_attrs: " + str(m_attrs))
    print("x_targets: " + str(x_targets))
    print("y_targets: " + str(y_targets))
    p_value = weat_rand_test(wv, x_targets, y_targets, m_attrs, f_attrs, 1000)
    cohens_d = get_cohens_d(wv,  x_targets, y_targets, m_attrs, f_attrs)
    print("{}\t{}\t{}\n".format("strength vs. weakness", p_value, cohens_d))


def filter_words(wv, words):
    final_words = []
    for word in words:
        if word in wv.vocab:
            final_words.append(word)
    return final_words


def weat_rand_test(wv, m_words, f_words, m_attrs, f_attrs, iterations):
    u_words = m_words + f_words
    runs = np.min((iterations, math.factorial(len(u_words))))
    seen = set()

    original = test_statistic(wv, m_words, f_words, m_attrs, f_attrs)
    r = 0
    for _ in range(runs):
        permutation = tuple(random.sample(u_words, len(u_words)))
        if permutation not in seen:
            m_hat = permutation[0:len(m_words)]
            f_hat = permutation[len(f_words):]
            if test_statistic(wv, m_hat, f_hat, m_attrs, f_attrs) > original:
                r += 1
            seen.add(permutation)
    p_value = r / runs
    return p_value


def get_cohens_d(wv,  m_targets, f_targets, m_attrs, f_attrs):
    if len(m_targets) == 0 or len(f_targets) == 0:
        return "NA"
    m_sum, f_sum = test_sums(wv, m_targets, f_targets, m_attrs, f_attrs)
    m_mean = m_sum / len(m_targets)
    f_mean = f_sum / len(f_targets)
    m_u_f = np.array([cosine_means_difference(wv, w, m_attrs, f_attrs) for w in m_targets + f_targets])
    stdev = m_u_f.std(ddof=1)
    return (m_mean - f_mean) / stdev


def test_statistic(wv, m_targets, f_targets, m_attrs, f_attrs):
    m_sum, f_sum = test_sums(wv, m_targets, f_targets, m_attrs, f_attrs)
    return m_sum - f_sum


def test_sums(wv, m_targets, f_targets, m_attrs, f_attrs):
    m_sum = 0.0
    f_sum = 0.0
    for t in m_targets:
        m_sum += cosine_means_difference(wv, t, m_attrs, f_attrs)
    for t in f_targets:
        f_sum += cosine_means_difference(wv, t, m_attrs, f_attrs)
    return m_sum, f_sum


def cosine_means_difference(wv, word, male_attrs, female_attrs):
    male_mean = cosine_mean(wv, word, male_attrs)
    female_mean = cosine_mean(wv, word, female_attrs)
    result = male_mean - female_mean
    #print("current word: " + word + "\tmale_mean: " + str(male_mean) + "\tfemale_mean: " + str(female_mean) + "\t result: " + str(result))
    return result


def cosine_mean(wv, word, attrs):
    return wv.cosine_similarities(wv[word], [wv[w] for w in attrs]).mean()


def get_cluster_description(wv, cluster_words):
    cluster_centroid = wv[cluster_words].sum(axis=0) / len(cluster_words)
    most_similars = wv.similar_by_vector(cluster_centroid.T, topn=10)
    words = [elem[0] for elem in most_similars]
    return ','.join(words)


def load_clusters(cluster_file, vocab):
    cluster2word = {}
    with open(cluster_file, 'r') as fclus:
        for line in fclus:
            fields = line.strip().split('\t')
            if len(fields) != 2:
                continue
            else:
                word = fields[0]
                cluster = int(fields[1])
            if word not in vocab:
                continue
            cluster_rec = cluster2word.get(cluster, None)
            if cluster_rec is None:
                cluster_rec = []
                cluster2word[cluster] = cluster_rec
            cluster_rec.append(word)
    return cluster2word


def choose_words(m_words, f_words, m_scores, f_scores, nwords=-1):
    n = np.min([len(m_words), len(f_words)] + ([nwords] if nwords >= 0 else []))
    m_scores = np.array(m_scores)
    f_scores = np.array(f_scores)
    m_topn_indices = m_scores.argsort()[::-1][:n]
    f_topn_indices = f_scores.argsort()[::-1][:n]
    sel_m_words = [m_words[i] for i in m_topn_indices]
    sel_f_words = [f_words[i] for i in f_topn_indices]
    return sel_m_words, sel_f_words


def maj_gender(genders):
    return 'F' if genders['F'] > genders['M'] else '=' if genders['M'] == genders['F'] else 'M'


def print_line(f, cluster, majority_gender, cluster_desc, m_words, f_words, p_value, cohens_d):
    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(cluster, majority_gender, cluster_desc, ",".join(m_words),
                                                  ",".join(f_words), p_value, cohens_d))

def main():
    #model = load("model", "cbt_train.txt", "w2v")
    # model = load("data/gap-full.bin", "none", "ft")
    model = ModelHandler.create_and_load().model

    m = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son', 'father', 'uncle', 'grandfather']
    f = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter', 'mother', 'aunt', 'grandmother']
    x = ['power', 'strong', 'confident', 'dominant', 'potent', 'command', 'assert', 'loud', 'bold', 'succeed',
         'triumph', 'leader', 'shout', 'dynamic', 'winner']
    y = ['weak', 'surrender', 'timid', 'vulnerable', 'weakness', 'wispy', 'withdraw', 'yield', 'failure', 'shy',
         'follow', 'lose', 'fragile', 'afraid', 'loser']

    gender_bias_test(x, y, m, f, model.wv)

    print(format(model.wv.most_similar(positive="girl", topn=10)))
    print(format(model.wv.similarity('queen', 'weak')))


    mbk = MiniBatchKMeans(init='k-means++', n_clusters=100, batch_size=100, max_no_improvement=10, verbose=0)
    mbk.fit(model.wv.vectors)

    with open("clustering.txt", 'w') as fout:
        for word, cluster in zip(model.wv.vocab, mbk.labels_):
            fout.write("{}\t{}\n".format(word, cluster))
    print("Done!")

    cluster2word = load_clusters("clustering.txt", model.wv.vocab)
    with open("clustering_sorted", 'w') as fout:
        for cluster in sorted(cluster2word.keys()):
            cluster_words = cluster2word[cluster]
            cluster_desc = get_cluster_description(model.wv, cluster_words)
            for word in cluster_words:
                score = cosine_means_difference(model.wv, word, m, f)
                gender = 'F' if score < 0 else 'M'
                fout.write("{}\t{}\t{}\t{}\t{}\n".format(cluster, cluster_desc, word, gender, score))
    print("Done!")

    last_cluster = None
    f_words = []
    f_scores = []
    m_words = []
    m_scores = []
    genders = {'F': 0, 'M': 0}
    with open("clustering_sorted", 'r') as fga, open("clusering_end", 'w') as fout:
        for lgassoc in fga:
            fields = lgassoc.strip().split('\t')
            cluster = int(fields[0])
            cluster_desc = fields[1]
            word = fields[2]
            gender_score = float(fields[4])
            if gender_score < 0:
                f_words.append(word)
                f_scores.append(gender_score * -1)  # we convert it to a positive value
                genders['F'] += 1
            else:
                m_words.append(word)
                m_scores.append(gender_score)
                genders['M'] += 1
            if last_cluster != cluster and last_cluster is not None:
                m_words, f_words = choose_words(m_words, f_words, m_scores, f_scores, -1)
                p_value = weat_rand_test(model.wv, m_words, f_words, m, f, 1000) if not False else "?"
                cohens_d = get_cohens_d(model.wv,  m_words, f_words, m, f)
                print_line(fout, cluster, maj_gender(genders), cluster_desc, m_words, f_words, p_value, cohens_d)
                f_words = []
                m_words = []
                f_scores = []
                m_scores = []
                genders = {'F': 0, 'M': 0}
            last_cluster = cluster
        m_words, f_words = choose_words(m_words, f_words, m_scores, f_scores, -1)
        p_value = weat_rand_test(model.wv, m_words, f_words, m, f, 1000) if not False else "?"
        cohens_d = get_cohens_d(model.wv,  m_words, f_words, m, f)
        print_line(fout, cluster, maj_gender(genders), cluster_desc, m_words, f_words, p_value, cohens_d)
    print("Done! 3")


if __name__ == '__main__':
    main()

