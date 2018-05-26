from datetime import datetime

import humanize
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from tabulate import tabulate


le_outcomes = LabelEncoder()

oversampled = pickle.load(open('../oversampled-labelencoded_test20180515105408.pkl', 'rb'))
oversampled['stop_outcome'] = le_outcomes.fit_transform(oversampled['stop_outcome'])

oversampled = shuffle(oversampled, random_state=0)
outcomes = oversampled.pop('stop_outcome')

# # Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(oversampled, 
                                                    outcomes, 
                                                    test_size=0.2, 
                                                    random_state=0)


params_rfc = {
#     'class_weight': None,
    'n_estimators': [10, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 19],
    'max_features': [None, 'sqrt', 'log2'],
#     'min_impurity_split': [0.0000001],

#     'min_samples_split': [2, 5, 10], 
#     'min_samples_leaf':[1, 2, 4],

#     'min_weight_fraction_leaf': [0],

#     'max_leaf_nodes': [None, 100, 1000, 2000],
    'n_jobs': [8],
    'random_state': [0],
    'verbose': [3],
}

rfc = RandomForestClassifier()

clf = RandomizedSearchCV(rfc, params_rfc, scoring='accuracy', n_jobs=8, cv=5, verbose=3)
clf.fit(X_train, y_train)

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
print("timestamp = {}".format(timestamp))

output_file = '{}-randomforestclassifier-clf.pkl'.format(timestamp)
pickle.dump(clf, open(output_file, 'wb'))
print('Saved {}'.format(output_file))