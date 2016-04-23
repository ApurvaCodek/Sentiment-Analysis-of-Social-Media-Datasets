"""SEntiment analysis
Ver 1.0"""

import sys
import csv
import pdb
import string
from random import shuffle
import collections

import nltk, nltk.classify.util, nltk.metrics
from nltk.collocations import *
from nltk.classify import NaiveBayesClassifier
from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support as pr
from sklearn.metrics import classification_report

from string import punctuation

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
    
def split_list(a_list):
    half = len(a_list)/2
    return a_list[:half], a_list[half:]

def remove_stoppers(sentence,stopset):
    querywords  = sentence.split()
    resultwords = sentence.split()
    stopper     = stopset.split() 
    for word in querywords:
        if word in stopper:
            resultwords.remove(word)
        sentence = ' '.join(resultwords)
    return sentence
    
def get_words_in_tweets(tweets):
	all_words = []
	for (words,sentiment) in tweets:
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
    
def bigramReturner (tweetString):
    tweetString = tweetString.lower()
    #remove punctuation
    tweetString = strip_punctuation(tweetString)
    bigramFeatureVector = []
    for item in nltk.bigrams(tweetString.split()):
        bigramFeatureVector.append(' '.join(item))
    return bigramFeatureVector

def trigramReturner(tweetString):
    tweetString = tweetString.lower()
    trigramFeatureVector = []
    for item in nltk.ngrams(tweetString.split(), 3):
        trigramFeatureVector.append(' '.join(item))
    return trigramFeatureVector

data,bigram_data, trigram_data = [],[],[]
train_data = []
test_data = []
predicted = []
observed  = []
train_data_bi, train_data_tri = [],[]
test_data_bi, test_data_tri = [],[]
predicted_bi, predicted_tri = [],[]
observed_bi, observed_tri  = [],[]

#stopset = set(stopwords.words('english'))
with open('minimal-stop.txt', 'r') as myfile:
    stopset = myfile.read().replace('\n', ' ')
    
#training corpus has the elements: tweet date/tweet id/sentiment/tweet text

f = csv.reader(open("trainBfull.tsv", "rU"), delimiter = '\t', skipinitialspace = True)
flist = list(f)
tweets = iter(flist)

countPos = 0
countNeg = 0
countNtr = 0
countObj = 0

next(tweets)
for flines in tweets:
    sentence = flines[3]
    sentiment = flines[2]
    sentence = strip_punctuation(sentence)
    sentence = remove_stoppers(sentence,stopset)
    
    bigram_sentence = bigramReturner(sentence)
    trigram_sentence = trigramReturner(sentence)
    sentence = [e.lower() for e in sentence.split() if len(e) >= 3]
    if sentiment == 'objective-OR-neutral':
        sentiment = 'objective'
    data.append((sentence, sentiment))
    bigram_data.append((bigram_sentence,sentiment))
    trigram_data.append((trigram_sentence,sentiment))

c = list(zip(data, bigram_data, trigram_data))
shuffle(c)
data, bigram_data, trigram_data = zip(*c)

train_data, test_data = split_list(data)
print 'training and test data has been constructed'
print 'training data length:%d' %len(train_data)
print 'test data length:%d' %len(test_data)

for (sentence,sentiment) in test_data:
    if sentiment == 'positive':
        countPos = countPos + 1
    elif sentiment == 'negative':
        countNeg = countNeg + 1
    elif sentiment == 'neutral':
        countNtr = countNtr + 1
    elif sentiment == 'objective':
        countObj = countObj + 1
    elif sentiment == 'objective-OR-neutral':
        sentiment = 'objective'
    countObj  = countObj + 1
print 'test data stats'
print 'positive  sentiment:%d' %countPos
print 'negative  sentiment:%d' %countNeg
print 'neutral   sentiment:%d' %countNtr
print 'objective sentiment:%d' %countObj

word_features = get_word_features(get_words_in_tweets(train_data))
training_set = nltk.classify.apply_features(extract_features,train_data)
classifier = nltk.NaiveBayesClassifier.train(training_set);
print 'classifier defined'


for items in test_data:
	sentence = ' '.join(items[0])
	predicted.append(classifier.classify(extract_features(sentence.split())))
	observed.append(items[1]) 
	
#print word_features
classifier.show_most_informative_features(5)
print(classification_report(observed, predicted, target_names=['negative','neutral','objective','positive']))

print 'Confusion Matrix'
print nltk.ConfusionMatrix( observed, predicted )



#####bigarm features

train_data_bi, test_data_bi = split_list(bigram_data)
print 'training and test data has been constructed for bigram'
print 'training data length:%d' %len(train_data_bi)
print 'test data length:%d' %len(test_data_bi)


word_features = get_word_features(get_words_in_tweets(train_data_bi))
training_set = nltk.classify.apply_features(extract_features,train_data_bi)
classifier = nltk.NaiveBayesClassifier.train(training_set);
print 'classifier defined'


for items in test_data_bi:
    sentence = ' '.join(items[0])
    sentence = bigramReturner(sentence)
    predicted_bi.append(classifier.classify(extract_features(sentence)))
    observed_bi.append(items[1]) 
	
#print word_features
classifier.show_most_informative_features(5)
print(classification_report(observed_bi, predicted_bi, target_names=['negative','neutral','objective','positive']))

print 'Confusion Matrix'
print nltk.ConfusionMatrix( observed_bi, predicted_bi )



#####trigram features

train_data_tri, test_data_tri = split_list(trigram_data)
print 'training and test data has been constructed for trigram'
print 'training data length:%d' %len(train_data_tri)
print 'test data length:%d' %len(test_data_tri)


word_features = get_word_features(get_words_in_tweets(train_data_tri))
training_set = nltk.classify.apply_features(extract_features,train_data_tri)
classifier = nltk.NaiveBayesClassifier.train(training_set);
print 'classifier defined'


for items in test_data_tri:
    sentence = ' '.join(items[0])
    sentence = trigramReturner(sentence)
    predicted_tri.append(classifier.classify(extract_features(sentence)))
    observed_tri.append(items[1]) 
	
#print word_features
classifier.show_most_informative_features(5)
print(classification_report(observed_tri, predicted_tri, target_names=['negative','neutral','objective','positive']))

print 'Confusion Matrix'
print nltk.ConfusionMatrix( observed_tri, predicted_tri)





