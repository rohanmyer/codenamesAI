from nltk.data import find
import gensim
import gensim.downloader

import csv
import numpy as np
from sklearn.cluster import KMeans

# import nltk
from nltk.corpus import wordnet as wn

# from nltk.corpus import brown

# word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
# model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=False)
model = gensim.downloader.load("glove-twitter-25")

# model = gensim.models.KeyedVectors.load_word2vec_format("vectorizers/numberbatch-en-19.08.txt", binary=False)

# try:
#     wn.ensure_loaded()
# except:
#     nltk.download('wordnet')
word_corpus = set(wn.words())
word_corpus = [x.lower() for x in word_corpus if x.isalpha()]

# word_corpus = []
# with open('wordlist.txt', 'r') as fd:
#     reader = csv.reader(fd)
#     for row in reader:
#         word_corpus.extend(row)


def guess(state) -> list:
    words = state.words
    guessed = state.guessed
    clue = state.clue.lower()
    words_scores = []
    for i, word in enumerate(words):
        if word == "new york":
            word = word.replace(" ", "-")
        else:
            word = word.replace(" ", "_")
        if guessed[i] == "UNKNOWN":
            words_scores.append((i, model.similarity(clue, word)))
    words_scores.sort(key=lambda x: x[1], reverse=True)
    return [words_scores[i][0] for i in range(state.count)]


def clue(state) -> tuple:
    words = state.words
    guessed = state.guessed
    actuals = state.actual

    for i, status in enumerate(guessed):
        if actuals[i] != state.color:
            continue
        if status == "UNKNOWN":
            word = words[i]
            break

    clue = ""
    sim = -1
    count = 0

    for i, w in enumerate(word_corpus):
        if "_" in w or " " in w:
            continue
        if w in word or word in w:
            continue
        if not w.isalpha():
            continue
        if w in words:
            continue
        close = []
        for j, word in enumerate(words):
            if word == "new york":
                word = word.replace(" ", "-")
            else:
                word = word.replace(" ", "_")
            try:
                close.append((j, model.similarity(w, word)))
            except:
                pass
        close.sort(key=lambda x: x[1], reverse=True)
        n = 0
        for c in close:
            if actuals[c[0]] == state.color and guessed[c[0]] == "UNKNOWN":
                n += 1
            else:
                break
        if n > count:
            count = n
            clue = w
            sim = close[0][1]
    return clue, min(count, 3)
