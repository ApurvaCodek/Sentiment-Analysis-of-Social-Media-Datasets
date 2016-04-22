"""SEntiment analysis
Ver 1.0"""

import os
import sys
import csv
import pdb
import time
from random import shuffle
import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.metrics import BigramAssocMeasures
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support as pr
from sklearn.metrics import classification_report

def split_list(a_list):
    half = len(a_list)/2
    return a_list[:100], a_list[100:200]

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

text_file = open("minimal-stop.txt", "r")
stoplines = text_file.readlines()
for i in stoplines:
	print i
f = csv.reader(open("full-corpus.csv", "rb"), delimiter = ',', skipinitialspace = True)
flist = list(f)
tweets = iter(flist)
next(tweets)
for lines in tweets:
	words = lines[4]
	sentiment = lines[1]
	words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
	# treat neutral and irrelevant the same
    	if sentiment == 'irrelevant':
        	sentiment = 'neutral'
	data.append((words_filtered, sentiment))
	
shuffle(data)
train_data, test_data = split_list(data)
print 'training and test data has been constructed'
print 'training data length:%d' %len(train_data)
print 'test data length:%d' %len(test_data)


word_features = get_word_features(get_words_in_tweets(train_data))
training_set = nltk.classify.apply_features(extract_features,train_data)
classifier = nltk.NaiveBayesClassifier.train(training_set);
print 'classifier defined'

predicted = []
observed  = []

for items in test_data:
	sentence = ' '.join(items[0])
	predicted.append(classifier.classify(extract_features(sentence.split())))
	observed.append(items[1]) 
	
#print word_features
precision, recall, fscore, support = pr(observed, predicted)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print ('Most Informative Features :')
classifier.show_most_informative_features(5)

print(classification_report(observed, predicted, target_names=['negative','neutral','positive']))

print 'Confusion Matrix'
print nltk.ConfusionMatrix( observed, predicted )




#fill out frequecy distributions, incrementing the counter of each word
#within the appropriate distribution
"""def create_word_scores(training_data):
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for items in training_data:
        if (items[1] == 'positive'):
	    for i in items[0]:
            	cond_word_fd['positive'].inc(items[0])
        elif(items[1] == 'negative'):
            cond_word_fd['negative'].inc(items[0])
        elif(items[1] == 'neutral'):
            cond_word_fd['neutral'].(items[0])
    pdb.set_trace()
    #count of words in positive negative and neutral
    pos_word_count = cond_word_fd['positive'].N()
    neg_word_count = cond_word_fd['negative'].N()
    ntr_word_count = cond_word_fd['neutral'].N()
    total_word_count = pos_word_count + neg_word_count + ntr_word_count

    #chi-squared test to score the words
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['positive'][word],
            (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['negative'][word],
            (freq, neg_word_count), total_word_count)
        ntr_score = BigramAssocMeasures.chi_sq(cond_word_fd['neutral'][word],
            (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score + ntr_score
    return word_scores

#finds the best words given a set of scores and an n:
def best_word_feats(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])"""


"""numbers_to_test = 1000
#tries the best_word_features mechanism with each of the numbers_to_test of features
print 'evaluating best %d word features' % (numbers_to_test)
word_scores = create_word_scores(train_data)
best_words = find_best_words(word_scores, numbers_to_test)
training_set = nltk.classify.apply_features(extract_features,train_data)
classifier = nltk.NaiveBayesClassifier.train(training_set)

predicted = []
observed  = []

for items in test_data:
        sentence = ' '.join(items[0])
        predicted.append(classifier.classify(extract_features(sentence.split())))
        observed.append(items[1]) 
                
#print word_features
precision, recall, fscore, support = pr(observed, predicted)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print ('Most Informative Features :')"""

