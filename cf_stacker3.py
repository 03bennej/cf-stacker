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


def model(W, H, mu, b1, b2):
    return tf.linalg.matmul(W, H) + mu + b1 + b2


def lr_model(X, W_lr, b_lr):
    X_tf = tf.constant(X, dtype=tf.dtypes.float32)
    return tf.add(tf.matmul(X_tf, W_lr), b_lr)


def calculate_biases(X):
    mu = np.mean(X)
    mu1 = np.expand_dims(np.mean(X, axis=0), axis=0)
    mu2 = np.expand_dims(np.mean(X, axis=1), axis=1)

    mu = tf.constant(mu, dtype=tf.dtypes.float32)
    b1 = tf.constant(mu1 - mu, dtype=tf.dtypes.float32)
    b2 = tf.constant(mu2 - mu, dtype=tf.dtypes.float32)
    return mu, b1, b2


def define_variables(X_shape, latent_dim):
    initializer = keras.initializers.RandomUniform(minval=0,
                                                   maxval=0.01,
                                                   seed=None)
    X1, X2 = X_shape
    W = tf.Variable(initializer(shape=[X1, latent_dim],
                                dtype=tf.dtypes.float32),
                    trainable=True)
    H = tf.Variable(initializer(shape=[latent_dim, X2],
                                dtype=tf.dtypes.float32),
                    trainable=True)
    W_lr = tf.Variable(initializer(shape=[X2, X2],
                                   dtype=tf.dtypes.float32),
                       trainable=True)
    b_lr = tf.Variable(initializer(shape=[1, X2],
                                   dtype=tf.dtypes.float32),
                       trainable=True)
    return W, H, W_lr, b_lr


def wmse(X_true, X_pred, weights=None):
    if weights is None:
        return tf.reduce_mean(tf.pow(X_true - X_pred, 2))
    else:
        return tf.reduce_mean(tf.math.multiply(weights, tf.pow(X_true - X_pred, 2)))


def l2_reg(U, lam):
    return lam * (tf.reduce_mean(tf.pow(U, 2)))


def optimize_W(X, W, H, C, mu, b1, b2, lam, optimizer):
    with tf.GradientTape() as tape:
        X_pred = model(W, H, mu, b1, b2)
        loss = wmse(X, X_pred, C) + l2_reg(W, lam) + l2_reg(H, lam)

    gradients = tape.gradient(loss, [W])

    optimizer.apply_gradients(zip(gradients, [W]))


def optimize_H(X, W, H, C, mu, b1, b2, lam, optimizer):
    with tf.GradientTape() as tape:
        X_pred = model(W, H, mu, b1, b2)
        loss = wmse(X, X_pred, C) + l2_reg(W, lam) + l2_reg(H, lam)

    gradients = tape.gradient(loss, [H])

    optimizer.apply_gradients(zip(gradients, [H]))


def optimize_C(X, C, W_lr, b_lr, optimizer):
    with tf.GradientTape() as tape:
        C_pred = lr_model(X, W_lr, b_lr)
        loss = wmse(C, C_pred, weights=None)

    gradients = tape.gradient(loss, [W_lr, b_lr])

    optimizer.apply_gradients(zip(gradients, [W_lr, b_lr]))


def optimization_step(X, W, H, C, mu, b1, b2, lam, W_lr, b_lr, optimizer):
    optimize_W(X, W, H, C, mu, b1, b2, lam, optimizer)

    optimize_H(X, W, H, C, mu, b1, b2, lam, optimizer)

    optimize_C(X, C, W_lr, b_lr, optimizer)


def optimize(X, W, H, C, mu, b1, b2, lam, W_lr, b_lr, optimizer, tol, max_iter,
             partial=False):
    step = 0

    X_tf = tf.constant(X, dtype=tf.dtypes.float32)
    C_tf = tf.constant(C, dtype=tf.dtypes.float32)

    X_pred = model(W, H, mu, b1, b2)
    C_pred = lr_model(X_tf, W_lr, b_lr)
    mf_loss = wmse(X_tf, X_pred, C_pred) + l2_reg(W, lam) + l2_reg(H, lam)
    conf_loss = wmse(C_tf, C_pred, weights=None)

    tot_loss = mf_loss + conf_loss

    while tot_loss > tol:

        if partial:

            optimize_W(X_tf, W, H, C_tf, mu, b1, b2, lam, optimizer)

        else:

            optimization_step(X_tf, W, H, C_tf, mu, b1, b2, lam, W_lr, b_lr, optimizer)

        step = step + 1

        if step % 50 == 0:
            X_pred = model(W, H, mu, b1, b2)
            C_pred = lr_model(X_tf, W_lr, b_lr)
            mf_loss = wmse(X_tf, X_pred, C_pred) + l2_reg(W, lam) + l2_reg(H, lam)
            conf_loss = wmse(C_tf, C_pred, weights=None)

            tot_loss = mf_loss + conf_loss

            print("epoch: %i, tot_loss: %f, mf_loss: %f, conf_loss: %f" % (step, tot_loss, mf_loss, conf_loss))

        if step == max_iter:
            print("Increase max_iter: unable to meet convergence criteria")
            break


class MatrixFactorizationClassifier(BaseEstimator):

    def __init__(self,
                 latent_dim=50,
                 C=1,
                 lam=0.0,
                 tol=0.0001,
                 max_iter=500,
                 learning_rate=0.1,
                 method="mean"):
        self.latent_dim = latent_dim
        self.C = C
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
        self.X_shape = np.shape(X)

        self.C = tf.constant(1 - np.abs(X - np.expand_dims(y, axis=1)),
                             dtype=tf.dtypes.float32)
        # self.C[self.C>=0.5] = 1
        # self.C[self.C<0.5] = 0

        self.W, self.H, self.W_lr, self.b_lr = define_variables(self.X_shape, self.latent_dim)

        self.mu, self.b1, self.b2 = calculate_biases(X)

        optimize(X, self.W, self.H, self.C, self.mu, self.b1, self.b2,
                 self.lam, self.W_lr, self.b_lr, self.optimizer,
                 self.tol, self.max_iter)

        return self

    def predict(self, X):
        self.X_shape = np.shape(X)

        self.W_predict, _, _, _ = define_variables(self.X_shape, self.latent_dim)

        self.C_predict = lr_model(X, self.W_lr, self.b_lr)

        self.mu, self.b1, self.b2_predict = calculate_biases(X)

        optimize(X, self.W_predict, self.H, self.C_predict, self.mu, self.b1, self.b2_predict,
                 self.lam, self.W_lr, self.b_lr, self.optimizer,
                 self.tol, self.max_iter, partial=True)

        self.X_predict = model(self.W_predict, self.H, self.mu, self.b1, self.b2_predict)

        self.parameters = {"W": self.W,
                           "W_predict": self.W_predict,
                           "H": self.H,
                           "mu": self.mu,
                           "b1": self.b1,
                           "b2": self.b2,
                           "b2_predict": self.b2_predict}

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

    X = X_train

    #
    mf_model = MatrixFactorization(latent_dim=1,
                                   max_iter=200,
                                   learning_rate=0.001,
                                   tol=0.01,
                                   lamW=0.0,
                                   lamH=0.0)

    initializer = keras.initializers.RandomUniform(minval=0,
                                                   maxval=1,
                                                   seed=None)

    mf_model.fit(X, labels_train)

    X_predict_probs, parameters = mf_model.predict(X_test)

    X_predict = np.copy(X_predict_probs)
    X_predict = np.round(X_predict_probs)
    # X_predict[X_predict >= 0.5] = 1
    # X_predict[X_predict < 0.5] = 0
    # np.integer(X_predict)

    X_predict_mean = np.mean(X_test, axis=1)
    X_predict_mean[X_predict_mean >= 0.5] = 1
    X_predict_mean[X_predict_mean < 0.5] = 0

    f1score_base = np.max(sklearn.metrics.f1_score(labels_test, X_predict_mean))

    f1score = np.max(sklearn.metrics.f1_score(labels_test, X_predict))

    acc_base = sklearn.metrics.accuracy_score(labels_test, X_predict_mean)

    accscore = sklearn.metrics.accuracy_score(labels_test, X_predict)

    print("f1 base: ", f1score_base)
    print("f1: ", f1score)

    print("acc base: ", acc_base)
    print("acc: ", accscore)
