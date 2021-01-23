import argparse
import json
import os

from bias_assessment_module.BiasAssessmentModule import BiasAssessmentModule


def parse_args():
    """ defines and parses arguments that the user passed on application start
    :return: arguments parsed
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-cr', '--corpus', type=str, choices=['CLLIP_Corpus', 'ChiLit_Corpus', 'CPBC_Corpus', 'GAP',
                                                              'GoogleNews'], help="Corpus name")
    parser.add_argument('-a', '--amount', type=int, help="Amount of sub-corpora: CLLIP_Corpus and СhiLit_Corpus can be "
                                                         "split in the smaller sub-corpora")
    parser.add_argument('-s', '--size', type=int, help="Dimensionality of the word vectors")
    parser.add_argument('-p', '--permutations', type=int, help="Number of permutations for a single WEAT")
    parser.add_argument('-w', '--window', type=int,
                        help="Maximum distance between the current and predicted word within a sentence")
    parser.add_argument('-sg', '--skipgram', type=int, choices=[1, 0],
                        help="Training algorithm: 1 for skip-gram; 0 CBOW")
    parser.add_argument('-m', '--mode', type=str, choices=['WEAT', 'Clustering', 'WEAT_and_Clustering'],
                        help="Running mode.")
    return parser.parse_args()


def main(args):
    """ Fetches arguments from user input, loads the main configuration from config.json file, merges
    the arguments from user input into the configuration, creates an instance of the BiasAssessmentModule,
    depending on the mode executes simple WEATs and/or WEATs on clusters. WEAT bias categories are defined
    inside this function.
    :param args: arguments specified by the user
    :return: None
    """
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    mode = config["mode"]
    model_config = config["model"]

    if args.corpus is not None:
        model_config["corpus_name"] = args.corpus
        model_config["corpus_path"] = "data" + os.path.sep + args.corpus
        model_config["model_path"] = "data" + os.path.sep + "cache" + os.path.sep + args.corpus
    if args.amount is not None: model_config["amount_of_corpora"] = args.amount
    if args.size is not None: model_config["size"] = args.size
    if args.permutations is not None: model_config["number_of_permutations"] = args.permutations
    if args.window is not None: model_config["window"] = args.window
    if args.skipgram is not None: model_config["sg"] = args.skipgram
    if args.mode is not None: mode = args.mode

    module = BiasAssessmentModule(config)

    if mode == "WEAT" or mode == "WEAT_and_Clustering":

        # comment out categories for which the WEAT should not be executed
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
        module.run_weat(bias_categories)

    if mode == "Clustering" or mode == "WEAT_and_Clustering":

        # comment out categories for which the WEAT should not be executed
        bias_categories_to_cluster = [
            "CG1_math_vs_reading",
            "CG2_math_vs_reading",
            "CA1_flowers_vs_insects"
        ]
        module.run_weat_with_clusters(bias_categories_to_cluster)


if __name__ == '__main__':
    main(parse_args())
