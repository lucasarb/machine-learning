from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import matplotlib
from nltk.stem import WordNetLemmatizer
from collections import Counter

#pre-processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.model_selection import train_test_split

#metrics
from sklearn.metrics import classification_report

#classifiers
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

#refinement
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier


def replace_similar_ingredients(i):
    #Removes or replaces extra words of the ingredients to have an uniform representation of the ingredients  
    
    i = i.replace('fresh ', '').replace(' fresh','').replace('diced ','')
    i = i.replace('minced ','').replace('chopped ','')
    i = i.replace('garlic cloves','garlic').replace('ground black pepper','black pepper').replace('large eggs','eggs')
    i = i.replace('extravirgin olive oil','olive oil').replace('fresh ginger','ginger').replace('allpurpose flour','flour')
    i = i.replace('vegetable oil', 'oil')
   
    return i


def clean_recipe(recipe):
    #Clean a single recipe. A single list of ingredients 

    letterRegex = re.compile('[^a-zA-Z ]')
    recipe = [letterRegex.sub('', i).strip() for i in recipe]

    recipe = [ingredient.lower() for ingredient in recipe]
    
    recipe = [replace_similar_ingredients(i) for i in recipe]
    
    recipe = [stemmer.lemmatize(i) for i in recipe]
    
    
    return recipe


def clean_data(train, test,stemmer):
    #Clean all the dataset and transform the ingredients from a list to a string

    train['source'] = 'train'
    test['source'] = 'test'

    data = pd.concat([train,test], ignore_index=True)

    data['ingredients_clean'] = data.ingredients.apply(lambda x: clean_recipe(x))
    data['ingredients_string'] = data.ingredients_clean.apply(lambda x: ' '.join(x))

    data.drop(['ingredients','ingredients_clean'], axis = 1, inplace=True)

    train = data.loc[data.source == 'train']
    test = data.loc[data.source == 'test']

    train.drop('source',axis = 1, inplace=True)
    test.drop(['source','cuisine'], axis= 1, inplace = True)

    #export the cleaned data to a new file
    train.to_csv('train_modified.csv', index=False)
    test.to_csv('test_modified.csv',index=False)

def train_classifier(clf,x_train,y_train):
    #Function that trains the classifiers that are going to be tested 

    start = time.time()
    clf.fit(x_train,y_train)
    end = time.time()

    print "Time for Training: {}".format(end - start)
    print "Train score: %.4f" % clf.score(x_train.toarray(),y_train)

    return clf

def train_test_score(clf,x_train,x_test,y_train,y_test,cuisine_list):
    #this functions measures the test score 

    clf = train_classifier(clf,x_train,y_train)

    print "Test score: %.4f" % clf.score(x_test.toarray(),y_test)

    pred = clf.predict(x_test)
    print "--------individual category report----------"
    print classification_report(y_test,pred, target_names = cuisine_list)
    print "-"*30

    return clf

def find_best_model(x_train,y_train):
    #Function that is responsible for the grid search and hyper parameter optimzation

    clf = LinearSVC()

    parameters = {'C':(0.5,1,0.1,0.01,0.001),'max_iter':(1000,2000,500,5000,10000)}

    search = GridSearchCV(estimator = clf,param_grid = parameters)

    search.fit(x_train,y_train)

    return search.best_estimator_
    

if __name__ == "__main__":
    """
        The program will import the datasets, clean it and save it in another file. After that, it trains and
        score all the classifiers that were cited in the report. Finally, the program does a grid search for the LinearSVC 
        and uses it as a bagging estimator.

    """
    train = pd.read_json('train.json')
    test = pd.read_json('test.json')

    stemmer = WordNetLemmatizer()

    clean_data(train, test,stemmer)
    print "Data cleaned..."

    #read the cleaned data from the file generated by clean_data function
    data = pd.read_csv('train_modified.csv')


    number_classes = (len(pd.Series.unique(data['cuisine'])))
    lbl_enc = LabelEncoder() #label encoder to transform the cuisine names into numbers

    corpustr = data['ingredients_string']
    vectorizertr = TfidfVectorizer(stop_words='english', ngram_range = ( 1, 1),analyzer="word", 
                               max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, norm = 'l2')

    count_matrix = vectorizertr.fit_transform(corpustr)# create a count matrix with the TF-IDF

    targets = data['cuisine']
    targets = lbl_enc.fit_transform(targets)

    cuisine_list =[lbl_enc.inverse_transform(i) for i in xrange(20)]

    # Separate the training dataset into train and test
    x_train, x_test, y_train, y_test = train_test_split(count_matrix ,targets ,test_size = 0.33, random_state = 42)

    print "Initializing training..."

    clf_SVC = LinearSVC(random_state = 42)
    clf_DT = DecisionTreeClassifier(random_state = 42)
    clf_NB = MultinomialNB()
    clf_MLP = MLPClassifier(solver = 'sgd', alpha = 1e-2, hidden_layer_sizes=(60,50), random_state=42)

    print "Training SVM classifier..."
    clf_SVC = train_test_score(clf_SVC,x_train,x_test,y_train,y_test,cuisine_list)

    print "Training Decision Tree Classifier..."
    clf_DT = train_test_score(clf_DT,x_train,x_test,y_train,y_test,cuisine_list)

    print "Training Naive Bayes Classifier..."
    clf_NB = train_test_score(clf_NB,x_train,x_test,y_train,y_test,cuisine_list)

    print "Training Multi-Layer Perceptron Classifier..."
    clf_MLP = train_test_score(clf_MLP,x_train,x_test,y_train,y_test, cuisine_list)

    print "Optimizing with grid search..."
    optim_clf = find_best_model(x_train,y_train)

    pred = optim_clf.predict(x_test)

    print "Training score after grid search: {}".format(optim_clf.score(x_test,y_test))
    print "--------individual category report----------"
    print classification_report(y_test,pred, target_names = cuisine_list)
    print "-"*30

    print "Using Bagging Classifier"
    bagging = BaggingClassifier(base_estimator = optim_clf, random_state = 42)
    bagging.fit(x_train,y_train)

    bag_score = bagging.score(x_test,y_test)
    pred = bagging.predict(x_test)

    print "Bagging Score: {}".format(bag_score)
    print "--------individual category report----------"
    print classification_report(y_test,pred, target_names = cuisine_list)
    print "-"*30
















