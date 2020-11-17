class BiasAssessor:
    def __init__(self, weat_attribute_words, weat_target_words):
        self._weat_attribute_words = weat_attribute_words
        self._weat_target_words = weat_target_words


    def create(self, weat_attribute_words, weat_target_words):
        bias_assessor = BiasAssessor(weat_attribute_words, weat_target_words)
        return BiasAssessor

    def gender_bias_test(model):
        f_attrs = filter_words(wv, f)
        m_attrs = filter_words(wv, m)
        x_targets = filter_words(wv, x)
        y_targets = filter_words(wv, y)
        print("f_attrs: " + str(f_attrs))
        print("m_attrs: " + str(m_attrs))
        print("x_targets: " + str(x_targets))
        print("y_targets: " + str(y_targets))
        p_value = weat_rand_test(wv, x_targets, y_targets, m_attrs, f_attrs, 1000)
        cohens_d = get_cohens_d(wv, x_targets, y_targets, m_attrs, f_attrs)
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

    def get_cohens_d(wv, m_targets, f_targets, m_attrs, f_attrs):
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
        # print("current word: " + word + "\tmale_mean: " + str(male_mean) + "\tfemale_mean: " + str(female_mean) + "\t result: " + str(result))
        return result

    def cosine_mean(wv, word, attrs):
        return wv.cosine_similarities(wv[word], [wv[w] for w in attrs]).mean()