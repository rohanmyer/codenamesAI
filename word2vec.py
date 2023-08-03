from nltk.data import find
import gensim
import gensim.downloader

import csv
import numpy as np
from sklearn.cluster import KMeans

# import nltk
from nltk.corpus import wordnet as wn

# from nltk.corpus import brown

word2vec_sample = str(find("models/word2vec_sample/pruned.word2vec.txt"))
# # model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
# # model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=False)
# model = gensim.downloader.load("glove-twitter-25")

# # model = gensim.models.KeyedVectors.load_word2vec_format("vectorizers/numberbatch-en-19.08.txt", binary=False)

# # try:
# #     wn.ensure_loaded()
# # except:
# #     nltk.download('wordnet')
# wordnet = set(wn.words())
# # word_corpus = [x.lower() for x in word_corpus if x.isalpha()]

# # word_corpus = []
# # with open('wordlist.txt', 'r') as fd:
# #     reader = csv.reader(fd)
# #     for row in reader:
# #         word_corpus.extend(row)

# word_corpus = []
# with open(word2vec_sample, "r") as fd:
#     reader = csv.reader(fd)
#     for row in reader:
#         row = row[0].split()[0].lower()
#         if row in wordnet:
#             word_corpus.append(row)
# word_corpus = word_corpus[1:]


class CodenamesPlayer:
    def __init__(self):
        self.clues_given = []
        self.model = gensim.downloader.load("glove-twitter-50")
        self.corpus = self.build_corpus()

    def build_corpus(self):
        word2vec_sample = str(find("models/word2vec_sample/pruned.word2vec.txt"))
        wordnet = set(wn.words())
        # return wordnet
        word_corpus = []
        with open(word2vec_sample, "r") as fd:
            reader = csv.reader(fd)
            for row in reader:
                row = row[0].split()[0].lower()
                if row in wordnet:
                    word_corpus.append(row)
        return word_corpus[1:]

    def guess(self, state) -> list:
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
                words_scores.append((i, self.model.similarity(clue, word)))
        words_scores.sort(key=lambda x: x[1], reverse=True)
        return [words_scores[i][0] for i in range(state.count)]

    def clue(self, state) -> tuple:
        words = state.words
        guessed = state.guessed
        actuals = state.actual

        clue = ""
        sim = -1
        count = 0

        for i, w in enumerate(self.corpus):
            if "_" in w or " " in w:
                continue
            if not w.isalpha():
                continue
            if w in words:
                continue
            close = []
            for j, word in enumerate(words):
                if w in word or word in w:
                    self.clues_given.append(w)
                if word == "new york":
                    word = word.replace(" ", "-")
                else:
                    word = word.replace(" ", "_")
                try:
                    close.append((j, self.model.similarity(w, word)))
                except:
                    pass
            close.sort(key=lambda x: x[1], reverse=True)
            n = 0
            assasin_idx = 1
            for i, c in enumerate(close):
                if (
                    actuals[c[0]] == state.color
                    and guessed[c[0]] == "UNKNOWN"
                    # and c[1] > 0.5
                ):
                    n += 1
                elif actuals[c[0]] == "ASSASIN":
                    assasin_idx = i
                else:
                    break
            avg_sim = np.mean([x[1] for x in close[: n + 1]])
            n *= avg_sim * np.sqrt(assasin_idx)
            if n > sim and w not in self.clues_given:
                sim = n
                count = n / (avg_sim * np.sqrt(assasin_idx))
                clue = w
        self.clues_given.append(clue)
        return clue, min(count, 3)
