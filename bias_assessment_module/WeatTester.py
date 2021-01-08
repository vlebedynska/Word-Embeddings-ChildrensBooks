import argparse
import json

from bias_assessment_module.BiasAssessmentModule import BiasAssessmentModule
from bias_assessment_module.EmbeddingsClusterer import EmbeddigsClusterer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--corpus', type=str, choices=['CLLIP_Corpus', 'ft'], default='w2v', help="Corpus name")
    return parser.parse_args()



if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('-c', '--corpus', type=str, default='w2v', help="Corpus name")
    # parser.parse_args()

    # if args.corpus == "w2v":
    #     with open(config, "r") as jsonFile:
    #         data = json.load(jsonFile)
    #     data["model"]["corpus_name"] = args.corpus
    #     with open(config, "w") as jsonFile:
    #         json.dump(data, jsonFile)

    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    module = BiasAssessmentModule(config)
    bias_categories = [
        "G1_career_vs_family",
        "G2_maths_vs_arts",
        "G3_science_vs_arts",
        "G4_intelligence_vs_appearance",
        "G5_strength_vs_weakness",
        "RL1_Christianity_vs_Islam",
        "RL2_Christianity_vs_Judaism",
        "RL3_Judaism_vs_Islam",
        "AG1_young_vs_old",
        "A1_flowers_vs_insects",
        "A2_innocent_sheep_vs_cruel_wolf",
        "A3_naive_bird_vs_clever_fox",
        "A4_strong_lion_vs_tender_mouse",
        "A5_faithful_dog_vs_selfish_cat",
        "CR1_European_American_vs_African_American",
        "CG1_math_vs_reading",
        "CG2_math_vs_reading",
        "CA1_flowers_vs_insects"
        ]

    bias_categories_to_cluster = [
        "G1_career_vs_family"
        ]

    module.run_weat(bias_categories)

    # module.run_weat_with_clusters(bias_categories_to_cluster)


