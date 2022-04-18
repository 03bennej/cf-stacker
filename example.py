#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:44:02 2022

@author: jamie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from nmf import NMF, _safe_mean
from cf_stacker import remove_unreliable_entries, evaluate

#%% Determine unreliable predictions
train_data = pd.read_csv('validation-10000.csv.gz')
test_data = pd.read_csv('predictions-10000.csv.gz')
train_data = train_data.drop(["id"], axis=1)
test_data = test_data.drop(["id"], axis=1)
X_train = train_data.drop(["label"], axis=1).to_numpy()
labels_train = train_data.pop("label").to_numpy()
y_train = np.round(np.abs(X_train - np.expand_dims(labels_train, axis=1)))
X_test = test_data.drop(["label"], axis=1).to_numpy()
labels_test = test_data.pop("label").to_numpy()
y_test = np.round(np.abs(X_test - np.expand_dims(labels_test, axis=1)))

shape = X_train.shape[1]

model = MultiOutputClassifier(estimator=LogisticRegression())
model.fit(X_train, y_train)

predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

X_train_new = remove_unreliable_entries(np.copy(X_train), predict_train)
X_test_new = remove_unreliable_entries(np.copy(X_test), predict_test)
#%% Collaborative filtering - NMF

# Training data
n_patients_train, n_predictors_train = X_train_new.shape
n_components=5
W_train = np.random.rand(n_patients_train, n_components) # / np.sqrt(_safe_mean(X)/ n_components)
H = np.random.rand(n_components, n_predictors_train) # / np.sqrt(_safe_mean(X)/ n_components)

model_train = NMF(n_components=n_components, 
                  max_iter = 500,
                  init='custom', 
                  solver='mu', 
                  alpha=1,
                  update_H=True)
W_train = model_train.fit_transform(X_train_new, W=W_train, H=H)
H = model_train.components_
H_check = np.copy(H)
X_train_new_impute = W_train @ H
print(X_train_new_impute)


# Test data
n_patients_test, n_predictors_test = X_test_new.shape
W_test = np.random.rand(n_patients_test, n_components)# / np.sqrt(_safe_mean(X)/ n_components)
model_test = NMF(n_components=n_components, 
                  max_iter = 500,
                  init='custom', 
                  solver='mu', 
                  alpha=1,
                  update_H=False) # stops update of H
W_test = model_test.fit_transform(X_test_new, W=W_test, H=H)
H = model_test.components_
X_test_new_impute = W_test @ H