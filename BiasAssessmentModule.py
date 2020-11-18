from BiasAssessor import BiasAssessor
from ModelHandler import ModelHandler
import json


class BiasAssessmentModule():

    def run(self):
        with open("config.json", "r") as config_file:
            config = json.load(config_file)
        model = ModelHandler.create_and_load(config["model"]).model
        bias_assessor = BiasAssessor.create(model, config["weat_lists"])
        bias_assessor.bias_test("gender")
        print(format(model.wv.most_similar(positive="girl", topn=10)))
        print(format(model.wv.similarity('queen', 'weak')))



if __name__ == '__main__':
    module = BiasAssessmentModule()
    module.run()
