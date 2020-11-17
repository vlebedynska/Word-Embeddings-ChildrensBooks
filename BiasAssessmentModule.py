from BiasAssessor import BiasAssessor
from ModelHandler import ModelHandler


class BiasAssessmentModule():

    def run(self):
        model = self.create_or_load_model("model2", "data/CLLIP_Corpus", "w2v")
        model.assess_bias(model, "data/weat_lists/gender_attr_words", "data/weat_lists/gender_target_words")


    def create_or_load_model(self, model_name, corpus_path, model_type, force=False):
        # model = load("model", "cbt_train.txt", "w2v")
        # model = load("data/gap-full.bin", "none", "ft")
        return ModelHandler.create_and_load(model_name, corpus_path, model_type, force).model

    def assess_bias(model, weat_attribute_words, weat_target_words):
        bias_assessor = BiasAssessor.create(weat_attribute_words, weat_target_words)
        bias_assessor.gender_bias_test(model)

        # gender_bias_test(x, y, m, f, model.wv)

        print(format(model.wv.most_similar(positive="girl", topn=10)))
        print(format(model.wv.similarity('queen', 'weak')))


if __name__ == '__main__':
    module = BiasAssessmentModule()
    module.run()


