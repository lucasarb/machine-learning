from __future__ import division
import time
import pandas as pd
import numpy as np
import re
#Natural language processing library
from nltk.stem import WordNetLemmatizer

#Machine learning library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


train = pd.read_csv('train_modified.csv')
test = pd.read_csv('test_modified.csv')

#group the test and training dataset
data = pd.concat([train,test], ignore_index=True)

number_classes = (len(pd.Series.unique(data['cuisine'])))
lbl_enc = LabelEncoder()

corpus = data['ingredients_string']
vectorizer = TfidfVectorizer(stop_words='english', ngram_range = ( 1, 1),analyzer="word", 
                               max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, norm = 'l2')
#create a single count matrix 
count_matrix = vectorizer.fit_transform(corpus)

#divide it between the training set and testing set
count_matrix_train = count_matrix[:train.shape[0]]
count_matrix_test = count_matrix[train.shape[0]:]

targets = data['cuisine'].iloc[:39774]
targets = lbl_enc.fit_transform(targets)

x_train = count_matrix_train
y_train = targets

#Using the hyperparameters that were obtained from the whats_cooking.py grid search
clf_SVM = LinearSVC(C=0.5, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)

start = time.time()
clf_SVM.fit(x_train,y_train)
end = time.time()


print "Time for Training: {}".format(end - start)
print "SVM Train score: %.4f" % clf_SVM.score(x_train,y_train)

#transform the labels into the original names
predictions = clf_SVM.predict(count_matrix_test)
predictions = lbl_enc.inverse_transform(predictions)

#format the output to the submission testes on kaggle
test_output = test
test_output['cuisine'] = predictions
test_output = test_output.drop('ingredients_string',axis = 1)
test_output.to_csv('submission.csv',index=False)



