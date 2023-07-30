from nltk.data import find
import gensim
import gensim.downloader

import numpy as np
from sklearn.cluster import KMeans
from bertopic import BERTopic

# word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
# model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=False)
model = gensim.downloader.load('glove-twitter-25')

def guess(state) -> list:
    words = state.words
    guessed = state.guessed
    clue = state.clue.lower()
    words_scores = []
    for i, word in enumerate(words):
        if guessed[i] == "UNKNOWN":
            print(word)
            words_scores.append((i, model.similarity(clue, word)))
    words_scores.sort(key=lambda x: x[1], reverse=True)
    return [words_scores[i][0] for i in range(state.count)]

def find_optimal_clusters(data, max_clusters):
    distortions = []

    for num_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)  # Sum of squared distances to the nearest cluster center

    # Select the optimal number of clusters based on the Elbow point
    # (where the distortion value stops decreasing significantly)
    elbow_point = np.argmin(np.gradient(distortions))

    return elbow_point + 1

def clue(state) -> tuple:
    words = state.words
    guessed = state.guessed
    actuals = state.actual

    word_vects = []
    for i, word in enumerate(words):
        if guessed[i] == "UNKNOWN" and actuals[i] == state.color:
            word_vects.append((word, model[word]))
    
    num_clusters = find_optimal_clusters([x[1] for x in word_vects], len(word_vects)//2)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_model = kmeans.fit([x[1] for x in word_vects])
    clue_vect = kmeans_model.cluster_centers_[0]

    clue = model.similar_by_vector(clue_vect, topn=1)[0][0]
    clue_words = [x[0] for x in word_vects if kmeans_model.predict([x[1]])[0] == 0]
    return clue, len(clue_words)
