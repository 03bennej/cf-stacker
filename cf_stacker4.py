#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:37:28 2022

@author: jamie
"""

import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_curve

def fmax_score(y_pred, y_true, beta=1):
    # beta = 0 for precision, beta -> infinity for recall, beta=1 for harmonic mean
    precision, recall, threshold = precision_recall_curve(y_true, y_pred)
    fmeasure = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
    argmax = np.nanargmax(fmeasure)
    return fmeasure[argmax], precision[argmax], recall[argmax]


# def summary

def model(W, H, mu, bw, bh):
    return tf.linalg.matmul(W, H) #+ mu + bw + bh


def calculate_biases(X):
    mu = np.mean(X)
    muw = np.expand_dims(np.mean(X, axis=1), axis=1)
    muh = np.expand_dims(np.mean(X, axis=0), axis=0)

    mu = tf.constant(mu, dtype=tf.dtypes.float32)
    bw = tf.constant(muw - mu, dtype=tf.dtypes.float32)
    bh = tf.constant(muh - mu, dtype=tf.dtypes.float32)
    return mu, bw, bh


def define_variables(X_shape, latent_dim):
    initializer = keras.initializers.RandomUniform(minval=-0.01,
                                                   maxval=0.01,
                                                   seed=None)
    X1, X2 = X_shape
    W = tf.Variable(initializer(shape=[X1, latent_dim],
                                dtype=tf.dtypes.float32),
                    trainable=True)
    H = tf.Variable(initializer(shape=[latent_dim, X2],
                                dtype=tf.dtypes.float32),
                    trainable=True)
    return W, H


# def wmse(X_true, X_pred, C=1):
#     C = tf.constant(C, dtype=tf.dtypes.float32)
#     return tf.reduce_mean(tf.math.multiply(C, tf.pow(X_true - X_pred, 2)))

def wmse(X_true, X_pred, C=1):
    C = tf.constant(C, dtype=tf.dtypes.float32)
    se = tf.math.multiply(C, tf.pow(X_true - X_pred, 2))
    non_zero = tf.cast(se != 0, dtype=tf.dtypes.float32)
    return tf.reduce_sum(se) / tf.reduce_sum(non_zero)

def l2_reg(U, lam):
    return lam * (tf.reduce_mean(tf.pow(U, 2)))
    # return lam * (tf.reduce_mean(tf.math.abs(U)))


def optimize_W(X, W, H, mu, bw, bh, lam, optimizer, C):
    with tf.GradientTape() as tape:
        X_pred = model(W, H, mu, bw, bh)
        loss = wmse(X, X_pred, C) + l2_reg(W, lam) + l2_reg(H, lam) \
                + l2_reg(bw, lam) + l2_reg(bh, lam)

    gradients = tape.gradient(loss, [W])

    optimizer.apply_gradients(zip(gradients, [W]))


def optimize_H(X, W, H, mu, bw, bh, lam, optimizer, C):
    with tf.GradientTape() as tape:
        X_pred = model(W, H, mu, bw, bh)
        loss = wmse(X, X_pred, C) + l2_reg(W, lam) + l2_reg(H, lam) \
                + l2_reg(bw, lam) + l2_reg(bh, lam)

    gradients = tape.gradient(loss, [H])

    optimizer.apply_gradients(zip(gradients, [H]))


def optimization_step(X, W, H, mu, bw, bh, lam, optimizer, C):
    optimize_W(X, W, H, mu, bw, bh, lam, optimizer, C)

    optimize_H(X, W, H, mu, bw, bh, lam, optimizer, C)


def optimize(X, W, H, mu, bw, bh, lam, optimizer, C, tol, max_iter,
             train=False):
    step = 0

    X_tf = tf.constant(X, dtype=tf.dtypes.float32)

    X_pred = model(W, H, mu, bw, bh)
    loss = wmse(X_tf, X_pred, C) + l2_reg(W, lam) + l2_reg(H, lam) \
            + l2_reg(bw, lam) + l2_reg(bh, lam)


    while loss > tol:

        if train:

            optimization_step(X_tf, W, H, mu, bw, bh, lam, optimizer, C)

        else:

            optimize_W(X_tf, W, H, mu, bw, bh, lam, optimizer, C)

        step = step + 1

        if step % 100 == 0:
            X_pred = model(W, H, mu, bw, bh)
            loss = wmse(X_tf, X_pred, C) + l2_reg(W, lam) + l2_reg(H, lam)

            print("epoch: %i, loss: %f" % (step, loss))

        if step == max_iter:
            print("Increase max_iter: unable to meet convergence criteria")
            break


class MatrixFactorizationClassifier(BaseEstimator):

    def __init__(self,
                 latent_dim=50,
                 lam=0.0,
                 tol=0.0001,
                 max_iter=500,
                 learning_rate=0.1,
                 method="mean"):
        self.latent_dim = latent_dim
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.method = method
        self.optimizer = keras.optimizers.Adam(self.learning_rate)

        self.X_shape = None
        self.W = None
        self.H = None
        self.mu = None
        self.b1 = None
        self.b2 = None

    def fit(self, X, y):
        
        self.C_train_true = 1 - np.abs(X - np.expand_dims(y, axis=1))
        self.C_train_true[self.C_train_true >= 0.5] = 1
        self.C_train_true[self.C_train_true < 0.5] = 0

        self.lr_model = LinearRegression()
        
        self.lr_model.fit(X, y)

        self.X_train = X
        
        self.X_train_shape = np.shape(X)

        # self.W, self.H = define_variables(self.X_shape, self.latent_dim)

        # self.mu, self.bw, self.bh = calculate_biases(X)

        # optimize(X, self.W, self.H, self.mu, self.bw, self.bh,
        #          self.lam, self.optimizer, self.C_train_true,
        #          self.tol, self.max_iter, train=True)

        return self

    def predict(self, X):

        self.X_comb = np.concatenate((self.X_train, X), axis=0)
        
        self.y_predict = self.lr_model.predict(X)
        
        self.C_predict = 1 - np.abs(X - np.expand_dims(self.y_predict, axis=1))
        self.C_predict[self.C_predict >= 0.5] = 1
        self.C_predict[self.C_predict < 0.5] = 0.0

        self.C_comb = np.concatenate((self.C_train_true, self.C_predict), axis=0)
        
        self.X_comb_shape = np.shape(self.X_comb)

        self.W_comb, self.H_comb = define_variables(self.X_comb_shape, self.latent_dim)

        self.mu, self.bw, self.bh = calculate_biases(self.C_comb * self.X_comb)

        optimize(self.X_comb, self.W_comb, self.H_comb, self.mu, self.bw, self.bh,
                 self.lam, self.optimizer, self.C_comb,
                 self.tol, self.max_iter, train=True)

        self.X_comb_predict = model(self.W_comb, self.H_comb, self.mu, self.bw, self.bh)

        self.X_predict = self.X_comb_predict[self.X_train_shape[0]::, :]

        if self.method == 'mean':
            self.X_predict = np.mean(self.X_predict, axis=1)
        elif self.method == 'median':
            self.X_predict = np.median(self.X_predict, axis=1)

        self.X_predict = np.clip(self.X_predict,
                                 a_min=0,
                                 a_max=1)

        return self.X_predict


if __name__ == "__main__":
    train_data = pd.read_csv('validation-10000.csv.gz')
    test_data = pd.read_csv('predictions-10000.csv.gz')
    train_data = train_data.drop(["id"], axis=1)
    test_data = test_data.drop(["id"], axis=1)
    X_train = train_data.drop(["label"], axis=1).to_numpy()
    labels_train = train_data.pop("label").to_numpy()
    y_train = 1 - np.abs(X_train - np.expand_dims(labels_train, axis=1))
    # Values close to 1 indicate high confidence, values close to 0 indicate low confidence
    X_test = test_data.drop(["label"], axis=1).to_numpy()
    labels_test = test_data.pop("label").to_numpy()
    y_test = 1 - np.abs(X_test - np.expand_dims(labels_test, axis=1))

    mf_model = MatrixFactorizationClassifier(latent_dim=3,
                                             max_iter=150,
                                             learning_rate=0.001,
                                             tol=0.00001,
                                             lam=0.0000)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, labels_train)
    labels_pred=lr_model.predict(X_test)
    
    C_predict = 1 - np.abs(X_test - np.expand_dims(labels_pred, axis=1))

    # y_test[y_test>=0.5] = 1 
    # y_test[y_test<0.5] = 0    

    mf_model.fit(X_train, labels_train)

    X_predict_probs = mf_model.predict(X_test)



    f1score, pscore, rscore = fmax_score(X_predict_probs, labels_test)

    f1score_b, pscore_b, rscore_b = fmax_score(np.mean(X_test, axis=1), labels_test)


    print("f1 score: ", f1score, 
          "\nprecision score:", pscore,
          "\nrecall score:", rscore)

    print("f1 score: ", f1score_b, 
          "\nprecision score:", pscore_b,
          "\nrecall score:", rscore_b)
