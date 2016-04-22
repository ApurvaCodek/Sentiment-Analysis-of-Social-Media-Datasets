from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

import csv
trainingData = ()
with open("trainBfull.tsv") as file:
	tsvreader = csv.reader(file, delimiter="\t")
	for line in tsvreader:
		if line[3] != "Not Available": # Removing missing tweets from the data.
			trainingData += ((line[3].split(" "), line[2]),)
print len(trainingData)
train_subj_docs = trainingData[:7000]
test_subj_docs = trainingData[7000:]
training_docs = train_subj_docs
testing_docs = test_subj_docs

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=5)
print len(unigram_feats)

sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)


for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
	print('{0}: {1}'.format(key, value))
