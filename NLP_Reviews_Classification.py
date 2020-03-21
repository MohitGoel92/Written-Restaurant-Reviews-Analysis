# Natural Language Processing - Restaurant reviews classification

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset

# Tab seperated file
# The columns in the data are seperated by tabs
# quoting = 3, this ignores all double quotes within the dataset

ds = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
# Stemming is taking the root of the word (Loved/Loving => Love)

import re # Regular Expressions: Matching operations
import nltk # Natural Toolkit, Get rid of irrelevant words (the, that ... etc)
nltk.download('stopwords')
from nltk.corpus import stopwords # nltk.corpus - Read corpus files
from nltk.stem.porter import PorterStemmer # nltk.stem.porter - stem removes morphological affixes - porter for porter stemmer algorithm
ps = PorterStemmer() # Word stemmer based on porter stemmer algorithm

corpus = [] # The cleaned reviews
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', ds['Review'][i]) 
# Only keep the letters in the reviews by getting rid of numbers, punctuations etc 
# (do not remove characters that are either lower case a-z or upper case A-Z). Replace the rest with a space (single space)
# ds['Review'][i], dataset ds, Review Column, from row with index 0 to row with index 1000
    review = review.lower() # All capitals are replaced by lower case letters
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
# Stemming: ps.stem(word) - The root of the word
    review = ' '.join(review) # Join all the words but seperate them with a single space
    corpus.append(review)

# Creating the Bag of Words model
# The Bag of Words model involves two things: - A vocabulary of known words
#                                             - A measure of the presence of known words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # Choose 1500 of the most relevant words
X = cv.fit_transform(corpus).toarray() # Sparse matrix
y = ds.iloc[:,-1].values

# Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# We do not use feature scaling as the values are approximately between one and three (ish)

# Fitting the Naive Bayes model to the dataset

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

# Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)