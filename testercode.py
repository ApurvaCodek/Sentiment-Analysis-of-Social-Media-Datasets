"""SEntiment analysis
Ver 1.0"""

import os
import sys
import csv
import numpy as np
import nltk
import pdb
from random import shuffle
from sklearn.metrics import precision_recall_fscore_support as pr


train_tweets = [
    (['love this car'], 'positive'),
    (['this view amazing'], 'positive'),
    (['feel great this morning'], 'positive'),
    (['excited about the concert'], 'positive'),
    (['best friend'], 'positive'),
    (['not like this car'], 'negative'),
    (['this view horrible'], 'negative'),
    (['feel tired this morning'], 'negative'),
    (['not looking forward the concert'], 'negative'),
    (['enemy'], 'negative')]
test_tweets = [
    (['feel', 'happy', 'this', 'morning'], 'positive'),
    (['larry', 'friend'], 'positive'),
    (['not', 'like', 'that', 'man'], 'negative'),
    (['house', 'not', 'great'], 'negative'),
    (['your', 'song', 'annoying'], 'negative')]

def split_list(a_list):
    half = len(a_list)/2
    return a_list[:half], a_list[half:]

def get_words_in_tweets(tweets):
	all_words = []
	for (words, sentiment) in tweets:
		all_words.extend(words)
	return all_words

def get_word_features(wordlist):
	wordlist = nltk.FreqDist(wordlist)
	word_features = wordlist.keys()
	return word_features

def extract_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features
data = []
train_data = []
test_data = []

#training corpus has the elements: Topic/sentiment/tweet id/tweet date/tweet text
#tweets are shuffed before partitioining into training and test data

for lines in train_tweets:
    words = ' '.join(lines[0])
    sentiment = lines[1]
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    data.append((words_filtered, sentiment))


word_features = get_word_features(get_words_in_tweets(data))
training_set = nltk.classify.apply_features(extract_features, data)
classifier = nltk.NaiveBayesClassifier.train(training_set)

predicted = []
observed  = []

for items in test_tweets:
	tweets = items[0]
	sentence = ' '.join(tweets)
	predicted.append(classifier.classify(extract_features(sentence.split())))
	observed.append(items[1]) 
	
#print word_features
precision, recall, fscore, support = pr(observed, predicted)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))






