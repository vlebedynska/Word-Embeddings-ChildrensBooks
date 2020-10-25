import os
import gensim


def load():
    pass


def save():
    #https://radimrehurek.com/gensim/similarities/termsim.html
    pass


def main():
    text_file = open("cbt_train.txt", "r")
    documents = text_file.read().split('_BOOK_TITLE_')
    del documents[0]
    output_text = []
    for document in documents:
        output_text.append(gensim.utils.simple_preprocess(document))
    print(output_text[0])
    model = gensim.models.Word2Vec(output_text, size=150, window=10, min_count=2, workers=4)
    model.train(output_text, total_examples=len(output_text), epochs=10)

    m = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son', 'father', 'uncle', 'grandfather', 'prince', 'king']
    f = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter', 'mother', 'aunt', 'grandmother', 'queen', 'princess']
    x = ['power', 'strong', 'confident', 'dominant', 'potent', 'command', 'assert', 'loud', 'bold', 'succeed', 'triumph', 'leader', 'shout', 'dynamic', 'winner']
    y = ['weak', 'surrender', 'timid', 'vulnerable', 'weakness', 'wispy', 'withdraw', 'yield', 'failure', 'shy', 'follow', 'lose', 'fragile', 'afraid', 'loser']

    print(format(model.wv.most_similar(positive="king", topn=10)))
    print(format(model.wv.similarity('queen', 'weakness')))


if __name__ == '__main__':
    main()
