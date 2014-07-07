baseline-sentimentanalyser
==========================

A program to determine the sentiment of a review using Naive Bayes classifier trained on labeled data and maximum likelihood n-gram language model

    class BaselineSentimentAnalyser
     |  A program to determine the sentiment of a review using Naive Bayes classifier trained on labeled data and maximum likelihood n-gram language model.
     |  Usage:
     |  >>> bsa = BaselineSentimentAnalyser(['pos', 'neg'], 'documents/review_polarity/txt_sentoken/')
     |  >>> bsa.classify('documents/review.txt')
     |  pos
     |  
     |   Methods defined here:
     |  
     |   __init__(self, labels, location)
     |      Constructor Method to load training data to train Naive Bayes Classifier.
     |  
     |   calculateLikelihood(self, word, label)
     |      Method to calculate likelihood for a word.
     |  
     |  calculatePrior(self)
     |      Method to calculate Prior for labels.
     |  
     |  classify(self, document)
     |      Method to classify the document based on the training data.
     |  
     |  createUnigram(self)
     |      Method to create Unigram for training data.
     |  
     |  getDocuments(self, location)
     |      Method to get file names from training data.
     |  
     |  loadDocuments(self, loc, files)
     |      "Method to load training data.
     |  
     |  train(self)
     |      Method to train classifier and calculate Prior and Likelihood for words.
     |  
     |  unigramProbability(self, word, label)