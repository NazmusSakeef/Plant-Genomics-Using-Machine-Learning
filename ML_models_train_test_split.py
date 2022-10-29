# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:20:21 2022

@author: nazmu
"""

import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.decomposition import PCA

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
from numpy import mean
from numpy import std
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from tensorflow.keras.utils import to_categorical
# #y_train_bck = y_train
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#Cleaning the dataset, not necessary
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

#Input plant area values
ifile = pd.read_csv("D:/Plant Genomics Using ML/dataset/0_30_6class.csv") #Change it by using your own directory path


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))

X_col = ifile['timestamp']
ifile = ifile.drop('timestamp',axis = 1)
#sc = StandardScaler()
ifile.fillna(199, inplace=True)
sc = MinMaxScaler(feature_range = (0,1))

i_tr = np.transpose(ifile)
i_norm = []
i = 0
# for i in range(len(ifile)):
#     i_norm.append(stats.zscore((i_tr[[i]]))) 
for i in range(len(ifile)):
    i_norm.append(sc.fit_transform(i_tr[[i]]))
X_norm_a = i_norm
X_norm_a = np.asarray(X_norm_a) 
X_norm_a= np.reshape(X_norm_a, (X_norm_a.shape[0],X_norm_a.shape[1]))
#X_norm_a= np.reshape(X_norm_a, (46,1432))
X_norm_a = pd.DataFrame(X_norm_a)   

#Input corresponding labels
y = pd.read_csv("D:/Plant Genomics Using ML/dataset/0_30_6_class.csv") #Change it by using your own directory path
#y = pd.read_csv("D:/Research_work/exp4/0_min_6_class.csv") 

from sklearn.preprocessing import LabelEncoder

y_r = y

le = LabelEncoder()
y_r = le.fit_transform(y_r)
y_p = pd.DataFrame(y_r)
y_p.to_csv("D:/Research_work/Exp2/ResultZ1_diff_classes2/y_p9class8-9_2.csv")

X_norm_a['timestamp'] = X_col


#train_test split

X_train, X_test, y_train, y_test = train_test_split(X_norm_a,y_r, test_size=0.20, stratify=y)
    #X_train, X_test, y_train, y_test = train_test_split(ifile,y_r, test_size=0.20, stratify=y)
    #print(X_train.shape, y_train.shape)
    #print (X_test.shape, y_test.shape)
X_train_col = X_train['timestamp']    
X_train = X_train.drop('timestamp', axis =1)
X_test_col = X_test['timestamp']
X_test = X_test.drop('timestamp', axis =1)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
X_train.to_csv("D:/Research_work/Exp2/ResultZ1_diff_classes2/X_train_9class8-10_2.csv")
X_test.to_csv("D:/Research_work/Exp2/ResultZ1_diff_classes2/X_test_9class8-10_2.csv")
y_train.to_csv("D:/Research_work/Exp2/ResultZ1_diff_classes2/y_train_9class8-10_2.csv")
y_test.to_csv("D:/Research_work/Exp2/ResultZ1_diff_classes2/y_test_9class8-10_2.csv")
import csv
X_test_col.to_csv("D:/Research_work/Data/Result6_diff/X_test_reference_TM.csv")
X_train_col.to_csv("D:/Research_work/Data/Result6_diff/X_train_reference_TM.csv")

#converting to numpy array

X_train= np.asarray(X_train)
y_train=np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


#experimenting decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
#dt = DecisionTreeClassifier(random_state=3, max_depth=5)
for i in range(1,10,1):
  dt.fit(X_train,y_train)

# """Results of Decision tree
y_predict = dt.predict(X_test)
print(y_predict)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_predict)
acc = accuracy_score(y_test, y_predict, normalize=False)
print("y_predict", y_predict)
print("y_test:", y_test)
print(result)
print(acc)

 


# Test with RandomForest
# """

#RANDOM FOREST


# # #Create a Gaussian Classifier
#rf=RandomForestClassifier(n_estimators=100, random_state = 3)
rf=RandomForestClassifier(n_estimators=200,criterion='gini', max_depth=None, min_samples_leaf=20, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=True, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

# # # """Results of Random Forest Ensemble"""
rf.fit(X_train,y_train)
y_predict = rf.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_predict)
acc = accuracy_score(y_test, y_predict, normalize=False)
print("y_predict", y_predict)
print("y_test:", y_test)
print(result)
print(acc)


#Logistic Regression Model Implementation
#lr = LogisticRegression()
lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
# for i in range(1,20,1):
#   lr.fit(X_train, y_train)
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_predict)
acc = accuracy_score(y_test, y_predict, normalize=False)
print("y_predict", y_predict)
print("y_test:", y_test)
print(result)
print(acc)

# # Create first pipeline for base without reducing features.

pipe = Pipeline([('classifier' , RandomForestClassifier())])
# pipe = Pipeline([('classifier', RandomForestClassifier())])

# # Create param grid.

param_grid = [
    {'classifier' : [LogisticRegression()],
      'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(10,101,10)),
    'classifier__max_features' : list(range(6,32,5))}
      #'classifier__n_estimators' : list(range(10,2000,10)),
      #'classifier__max_features' : list(range(6,32,5)) )}
]

# # Create grid search object

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)



best_clf = clf.fit(X_train, y_train)
print(best_clf.best_params_)
y_predict = clf.predict(X_test)
print(y_predict)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_predict)
acc = accuracy_score(y_test, y_predict, normalize=False)
print("y_predict", y_predict)
print("y_test  :", y_test)
print(result)
print(acc)

  
# # # Create first pipeline for base without reducing features.

pipe = Pipeline([('classifier' , SVC())])
# pipe = Pipeline([('classifier', RandomForestClassifier())])
pipe = Pipeline([('classifier', LogisticRegression())])
# Create param grid.

param_grid = [
    {'classifier' : [LogisticRegression()],
      'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['newton-cg', 'lbfgs', 'liblinear'],},
    # {'classifier' : [RandomforestClassifier()],
    # 'classifier__n_estimators' : list(range(10,101,10)),
    # 'classifier__max_features' : list(range(6,32,5))}
      #'classifier__n_estimators' : list(range(10,2000,10)),
      #'classifier__max_features' : list(range(6,32,5)) )}
]

# Create grid search object

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

# Fit on data
param_grid = {'C': [0.1, 1, 10, 100],  
              
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              #'gamma':['scale', 'auto'],
              'kernel': ['poly','rbf','linear']}  
   
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,n_jobs=-1) 
   
# fitting the model for grid search 
grid.fit(X_train, y_train) 

clf.fit(X_train, y_train)
print(grid.best_params_)
print(clf.best_params_)

#best_clf = grid.fit(X_train, y_train)

y_predict = grid.predict(X_test)
#print(y_predict)
print(grid.best_params_)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_predict)
acc = accuracy_score(y_test, y_predict, normalize=False)
print("y_predict", y_predict)
print("y_test  :", y_test)
print(result)
print(acc)

#Implementation of SVM model with different Kerenel functionality

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
rbf = svm.SVC(kernel='rbf', gamma=1, C=10, decision_function_shape='ovo').fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=4, C=1, decision_function_shape='ovo').fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)
poly = svm.SVC(kernel='poly', C=1.0, degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None).fit(X_train,y_train)
linear_pred = linear.predict(X_test)
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
sig_pred = sig.predict(X_test)

accuracy_lin = linear.score(X_test, y_test)
accuracy_poly = poly.score(X_test, y_test)
accuracy_rbf = rbf.score(X_test, y_test)
accuracy_sig = sig.score(X_test, y_test)
print("Accuracy Linear Kernel:", accuracy_lin)
print("Accuracy Polynomial Kernel:", accuracy_poly)
print("Accuracy Radial Basis Kernel:", accuracy_rbf)
print("Accuracy Sigmoid Kernel:", accuracy_sig)

y_predict = poly.predict(X_test)
print(y_predict)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_predict)
acc = accuracy_score(y_test, y_predict, normalize=False)
print(result)
print(acc)

#BAGGING
pipeline = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None))
# pipeline = make_pipeline(StandardScaler(),
#                         SVC(kernel='poly', degree=4, C=1, decision_function_shape='ovo'))
#
# Instantiate the bagging classifier
#
# bg = BaggingClassifier(base_estimator=pipeline, n_estimators=100,
#                                   max_features=3,
#                                   #max_samples=100,
#                                   random_state=1, n_jobs=5)

bg = BaggingClassifier(base_estimator=pipeline, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
# for i in range(1,10,1):
#   bg.fit(X_train, y_train)
bg.fit(X_train,y_train)

y_predict = bg.predict(X_test)
print(y_predict)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_predict)
acc = accuracy_score(y_test, y_predict, normalize=False)
print(result)
print(acc)

# # #BOOSTING


# define dataset
#X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# evaluate the model
gb = GradientBoostingClassifier()
#gb=GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
#hb = HistGradientBoostingClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(gb, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
gb = GradientBoostingClassifier()
#hb = HistGradientBoostingClassifier()

#hb.fit(X_train, y_train)

for i in range(1,20,1):
  gb.fit(X_train, y_train)
gb.fit(X_train, y_train)
#"""Results of Boosting"""

y_predict = gb.predict(X_test)
print(y_predict)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_predict)
acc = accuracy_score(y_test, y_predict, normalize=False)
print(result)
print(acc)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
for i in range(1,10,1):
  kn.fit(X_train, y_train)
kn.fit(X_train, y_train)
y_predict = kn.predict(X_test)
print(y_predict)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_predict)
acc = accuracy_score(y_test, y_predict, normalize=False)
print(result)
print(acc)


#Stacking Model Implementation

 
#get the dataset
def get_dataset():
 	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
 	return X, y
 
#get a stacking ensemble of models
def get_stacking():

 	# define the base models
  pipeline = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None) )
  level0 = list()
  level0.append(('lr', LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)))
  #level0.append(('knn', KNeighborsClassifier(n_neighbors=3)))
  #level0.append(('cart', DecisionTreeClassifier()))
 
  #level0.append(('rf', RandomForestClassifier()))
  level0.append(('bg', BaggingClassifier(base_estimator=pipeline, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)))
  level0.append(('gb', GradientBoostingClassifier()))
  level0.append(('svm', SVC(kernel='poly', C=1.0, degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)))
  #level0.append(('bayes', GaussianNB()))
 	# define meta learner model
  level1 = SVC(kernel='poly', C=1.0, degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
 	# define the stacking ensemble
  model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
  return model
 
# # get a list of models to evaluate
def get_models():
  pipeline = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None))
  models = dict()
  models['lr'] = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
  #models['knn'] = KNeighborsClassifier()
  #models['cart'] = DecisionTreeClassifier()
  #models['rf'] = RandomForestClassifier()
  models['bg'] = BaggingClassifier(base_estimator=pipeline, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
  models['gb'] = GradientBoostingClassifier()
  models['svm'] = SVC(kernel='poly', C=1.0, degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
  #models['bayes'] = GaussianNB()
  models['stacking'] = get_stacking()
  return models
 
# # evaluate a give model using cross-validation
def evaluate_model(model, X_train, y_train):
 	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 	scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
 	return scores
 
# define dataset
#X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
 	scores = evaluate_model(model, X_train, y_train)
 	results.append(scores)
 	names.append(name)
 	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


#second modified ensemble
def get_dataset():
 	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
 	return X, y
 
#get a stacking ensemble of models
def get_stacking():

 	# define the base models
  pipeline = make_pipeline(StandardScaler(),
                             LogisticRegression(random_state=1))
  level0 = list()
  level0.append(('lr', LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)))
  #level0.append(('knn', KNeighborsClassifier(n_neighbors=3)))
  #level0.append(('cart', DecisionTreeClassifier()))
 
  #level0.append(('rf', RandomForestClassifier()))
  level0.append(('bg', BaggingClassifier(base_estimator=pipeline, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)))
  #level0.append(('gb', GradientBoostingClassifier()))
  level0.append(('svm', SVC(kernel='linear')))
  #level0.append(('bayes', GaussianNB()))
 	# define meta learner model
  level1 = SVC(kernel='linear')
 	# define the stacking ensemble
  model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
  return model
 
# # get a list of models to evaluate
def get_models():
  pipeline = make_pipeline(StandardScaler(),
                             LogisticRegression(random_state=1))
  models = dict()
  models['lr'] = LogisticRegression()
  #models['knn'] = KNeighborsClassifier()
  #models['cart'] = DecisionTreeClassifier()
  #models['rf'] = RandomForestClassifier()
  models['bg'] = BaggingClassifier(base_estimator=pipeline, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
  #models['gb'] = GradientBoostingClassifier()
  models['svm'] = SVC(kernel='poly')
  #models['bayes'] = GaussianNB()
  models['stacking'] = get_stacking()
  return models
 
# # evaluate a give model using cross-validation
def evaluate_model(model, X_train, y_train):
 	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 	scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
 	return scores
 
# define dataset
#X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
 	scores = evaluate_model(model, X_train, y_train)
 	results.append(scores)
 	names.append(name)
 	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()