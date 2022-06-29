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

def fmax_score(y_pred, y_true, beta=1, display=False):
    # beta = 0 for precision, beta -> infinity for recall, beta=1 for harmonic mean
    precision, recall, threshold = precision_recall_curve(y_true, y_pred)
    fmeasure = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
    argmax = np.nanargmax(fmeasure)
    
    f1score = fmeasure[argmax]
    pscore = precision[argmax]
    rscore = recall[argmax]
    
    if display:
        print("f1 score: ", f1score, 
              "\nprecision score:", pscore,
              "\nrecall score:", rscore)
    return f1score, pscore, rscore

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


def model(W, H, mu, bw, bh):
    return tf.linalg.matmul(W, H) #+ mu + bw + bh


def logistic_regression(X, W, b):
    return tf.nn.softmax(tf.matmul(X, W) + b)


def bce_loss(y_pred, y_true, W, b, lam):
    
    y_true = tf.one_hot(y_true, depth=2)

    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    
    bce = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))
    
    reg = l2_reg(W, lam) #+ l2_reg(b, lam)
    
    return bce + reg


def lr_optimization_step(X, y, W, b, optimizer, lam):

    with tf.GradientTape() as g:

        pred = logistic_regression(X, W, b)

        loss = bce_loss(pred, y, W, b, lam)

    gradients = g.gradient(loss, [W, b])

    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    return loss


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


def optimize_W(X, W, H, mu, bw, bh, lam, optimizer):
    with tf.GradientTape() as tape:
        X_pred = model(W, H, mu, bw, bh)
        loss = wmse(X, X_pred) + l2_reg(W, lam) + l2_reg(H, lam) \
                + l2_reg(bw, lam) + l2_reg(bh, lam)

    gradients = tape.gradient(loss, [W])

    optimizer.apply_gradients(zip(gradients, [W]))


def optimize_H(X, W, H, mu, bw, bh, lam, optimizer):
    with tf.GradientTape() as tape:
        X_pred = model(W, H, mu, bw, bh)
        loss = wmse(X, X_pred) + l2_reg(W, lam) + l2_reg(H, lam) \
                + l2_reg(bw, lam) + l2_reg(bh, lam)

    gradients = tape.gradient(loss, [H])

    optimizer.apply_gradients(zip(gradients, [H]))


def optimization_step(X, W, H, mu, bw, bh, lam, optimizer):
    optimize_W(X, W, H, mu, bw, bh, lam, optimizer)

    optimize_H(X, W, H, mu, bw, bh, lam, optimizer)


def optimize(X, W, H, mu, bw, bh, lam, optimizer, tol, max_iter,
             train=False):
    step = 0

    X_tf = tf.constant(X, dtype=tf.dtypes.float32)

    X_pred = model(W, H, mu, bw, bh)
    loss = wmse(X_tf, X_pred) + l2_reg(W, lam) + l2_reg(H, lam) \
            + l2_reg(bw, lam) + l2_reg(bh, lam)


    while loss > tol:

        if train:

            optimization_step(X_tf, W, H, mu, bw, bh, lam, optimizer)

        else:

            optimize_W(X_tf, W, H, mu, bw, bh, lam, optimizer)

        step = step + 1

        if step % 100 == 0:
            X_pred = model(W, H, mu, bw, bh)
            loss = wmse(X_tf, X_pred) + l2_reg(W, lam) + l2_reg(H, lam)

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

    def fit(self, X, y):

        self.fit_lr(X, y)

        return self

    def fit_mf(self, X, y):


    def fit_lr(self, X, y):
        
        X_tf = tf.constant(X, dtype=tf.dtypes.float32)

        self.W = tf.Variable(tf.zeros([X.shape[1], 2]), name="weight")
        self.b = tf.Variable(tf.zeros([2]), name="bias")
        
        y_hat = logistic_regression(X_tf, self.W, self.b)
        loss = bce_loss(y_hat, y, self.W, self.b, lam=self.lam).numpy()
        step=0
        while loss > self.tol:
            loss = lr_optimization_step(X_tf, 
                                        y, 
                                        self.W, 
                                        self.b, 
                                        self.optimizer,
                                        lam=self.lam).numpy()
            step+=1
            # if step % 100 == 0:
            #     print(loss)
            if step == self.max_iter:
                print("Increase max_iter: unable to meet convergence criteria")
                break
        return self


    def predict(self, X):
        X_tf = tf.constant(X, dtype=tf.dtypes.float32)
        self.y_pred = logistic_regression(X_tf, self.W, self.b).numpy()
        self.y_pred = self.y_pred[:,1]
        return self.y_pred


if __name__ == "__main__":
    train_data = pd.read_csv('validation-10000.csv.gz')
    test_data = pd.read_csv('predictions-10000.csv.gz')
    train_data = train_data.drop(["id"], axis=1)
    test_data = test_data.drop(["id"], axis=1)
    X_train = train_data.drop(["label"], axis=1).to_numpy()
    y_train = train_data.pop("label").to_numpy()
    # Values close to 1 indicate high confidence, values close to 0 indicate low confidence
    X_test = test_data.drop(["label"], axis=1).to_numpy()
    y_test = test_data.pop("label").to_numpy()

    mf_model = MatrixFactorizationClassifier(latent_dim=10,
                                             max_iter=200,
                                             learning_rate=0.1,
                                             tol=0.0000000001,
                                             lam=0.5)
    mf_model.fit(X_train, y_train)
    y_pred = mf_model.predict(X_test)
    
    sk_lr = LogisticRegression()
    sk_lr.fit(X_train, y_train)
    sk_y_pred = sk_lr.predict_proba(X_test)[:, 1]
    
    fmax_score(y_pred, y_test, display=True)
    fmax_score(sk_y_pred, y_test, display=True)

