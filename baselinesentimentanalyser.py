from __future__ import division
from collections import Counter
import os, re
import math as calc


class BaselineSentimentAnalyser():
    """A program to determine the sentiment of a review using Naive Bayes classifier trained on labeled data and maximum likelihood n-gram language model.
Usage:
>>> bsa = BaselineSentimentAnalyser(['pos', 'neg'], 'documents/review_polarity/txt_sentoken/')
>>> bsa.classify('documents/review.txt')
pos"""
    def __init__(self, labels, location):
        """Constructor Method to load training data to train Naive Bayes Classifier."""
        self.labels = labels
        self.files = self.getDocuments(location)
        self.words = self.loadDocuments(location, self.files)
        self.train()
        return 

    def getDocuments(self, location):
        """Method to get file names from training data."""
        labels = self.labels
        files = dict.fromkeys(labels, [])
        for label in labels:
            labelfile=[]
            for file in os.listdir(location+label+'/'):
                labelfile.append(file)
            files[label] = labelfile
        return files

    def loadDocuments(self, loc, files):
        """"Method to load training data."""
        words = dict.fromkeys(self.labels,"")
        for label in self.labels:
            for file in files[label]:
                handle = open(loc+label+'/'+file, 'r')
                words[label] = words[label] + ' ' + ' '.join(set(re.findall(r"<?/?\w+>?",handle.read().lower())))
                handle.close()
        for label in self.labels:
            words[label] = words[label].split()
        return words

    def train(self):
        """Method to train classifier and calculate Prior and Likelihood for words."""
        self.prior = self.calculatePrior()
        self.unigram = self.createUnigram()

    def calculatePrior(self):
        """Method to calculate Prior for labels."""
        prior = dict()
        for label in self.labels:
            prior[label] = len(self.files[label])
        s = sum(prior.values())
        for label in prior.keys():
            prior[label] = prior[label]/s
        return prior

    def createUnigram(self):
        """Method to create Unigram for training data."""
        unigram = dict.fromkeys(self.labels, dict())
        for label in self.labels:
            unigram[label] = Counter(self.words[label])
        return unigram

    def classify(self, document):
        """Method to classify the document based on the training data."""
        file = open(document, 'r')
        words = re.findall(r"<?/?\w+>?",file.read().lower())
        P = dict.fromkeys(self.labels, 0)
        for label in self.labels:
            for word in words:
                P[label] = P[label] + self.calculateLikelihood(word, label)
        P[label] = P[label] + calc.log(self.prior[label])
        print sorted(P, key=P.get, reverse=True)[0]

    def calculateLikelihood(self, word, label):
        """Method to calculate likelihood for a word."""
        return self.unigramProbability(word, label)

    def unigramProbability(self, word, label):
        """Method to calculate probability for input word."""
        return calc.log((self.unigram[label][word]+1)/(len(self.words[label])+len(self.unigram[label])))
