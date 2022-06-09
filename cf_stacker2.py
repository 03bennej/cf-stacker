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

def generate_mf_model(W, H, mu, b1, b2):
    return tf.linalg.matmul(W, H) + mu + b1 + b2


def model(W, H, mu, b1, b2):
    return tf.linalg.matmul(W, H) + mu + b1 + b2


def calculate_biases(X):
    mu = np.broadcast_to(np.mean(X), X.shape)
    mu1 = np.broadcast_to(np.mean(X, axis=0), X.shape)
    mu2 = np.transpose(
          np.broadcast_to(np.mean(X, axis=1), tuple(reversed(X.shape))))

    mu = tf.constant(mu, dtype = tf.dtypes.float32)
    b1 = tf.constant(mu1 - mu, dtype = tf.dtypes.float32)
    b2 = tf.constant(mu2 - mu, dtype = tf.dtypes.float32)
    return mu, b1, b2


def obj_fun(X_true, W, H, C, mu, b1, b2, lamW, lamH):
    #C[C >= 0.5] = 1
    #C[C < 0.5] = 0
    X_pred = model(W, H, mu, b1, b2)
    Cpow = tf.pow(tf.constant(C, dtype=tf.dtypes.float32), 2)
    Cpow = Cpow / tf.reduce_max(Cpow)
    wmse = tf.reduce_mean(tf.math.multiply(Cpow, tf.pow(X_true - X_pred, 2)))
    reg = lamW*tf.reduce_mean(tf.pow(W, 2)) + lamH*tf.reduce_mean(tf.pow(H, 2))
    return wmse + reg

def optimize_W(X, W, H, C, mu, b1, b2, lamW, lamH, optimizer):

    with tf.GradientTape() as tape:

        loss = obj_fun(X, W, H, C, mu, b1, b2, lamW, lamH)

    gradients = tape.gradient(loss, [W])

    optimizer.apply_gradients(zip(gradients, [W]))

def optimize_H(X, W, H, C, mu, b1, b2, lamW, lamH, optimizer):

    with tf.GradientTape() as tape:

        loss = obj_fun(X, W, H, C, mu, b1, b2, lamW, lamH)

    gradients = tape.gradient(loss, [H])

    optimizer.apply_gradients(zip(gradients, [H]))


def optimization_step(X, W, H, C, mu, b1, b2, lamW, lamH, optimizer):

    optimize_W(X, W, H, C, mu, b1, b2, lamW, lamH, optimizer)

    optimize_H(X, W, H, C, mu, b1, b2, lamW, lamH, optimizer)


def optimize(X, W, H, C, mu, b1, b2, lamW, lamH, optimizer, tol, max_iter,
             partial = False):

    step = 0

    X_tf = tf.constant(X, dtype=tf.dtypes.float32)

    loss = obj_fun(X_tf, W, H, C, mu, b1, b2, lamW, lamH)

    while (loss > tol):

        if partial:

            optimize_W(X_tf, W, H, C, mu, b1, b2, lamW, lamH, optimizer)

        else:

            optimization_step(X_tf, W, H, C, mu, b1, b2, lamW, lamH, optimizer)

        loss = obj_fun(X_tf, W, H, C, mu, b1, b2, lamW, lamH)

        step = step + 1

        if step % 50 == 0:

            print("epoch: %i, loss: %f" % (step, loss))

        if step == (max_iter):
            print("Increase max_iter: unable to meet convergence criteria")
            break


class MatrixFactorization(BaseEstimator):


    def __init__(self,
                 latent_dim=5,
                 C=1,
                 lamW=0.0,
                 lamH=0.0,
                 tol=0.0001,
                 max_iter=500,
                 learning_rate=0.1):

        self.latent_dim = latent_dim
        self.C = C
        self.lamW = lamW
        self.lamH = lamH
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.optimizer = keras.optimizers.Adam(self.learning_rate)
        self.initializer = keras.initializers.RandomUniform(minval=0,
                                                               maxval=1,
                                                               seed=None)

        self.X_shape = None
        self.W = None
        self.H = None
        self.mu = None
        self.b1 = None
        self.b2 = None

    def fit_transform(self, X):

        self.X_shape = np.shape(X)

        self.W = tf.Variable(self.initializer(shape=[self.X_shape[0],
                                                        self.latent_dim],
                                   dtype = tf.dtypes.float32),
                                   trainable = True)

        self.H = tf.Variable(self.initializer(shape = [self.latent_dim,
                                                  self.X_shape[1]],
                             dtype=tf.dtypes.float32),
                             trainable=True)

        self.mu, self.b1, self.b2 = calculate_biases(X)

        optimize(X, self.W, self.H, self.C, self.mu, self.b1, self.b2,
                 self.lamW, self.lamH, self.optimizer, self.tol, self.max_iter)

        return self


    def partial_fit_transform(self, X, H):

        self.X_shape = np.shape(X)

        self.W = tf.Variable(self.initializer(shape = [self.X_shape[0],
                                                        self.latent_dim],
                                   dtype = tf.dtypes.float32),
                                   trainable = True)

        self.H = H

        self.mu, self.b1, self.b2 = calculate_biases(X)

        optimize(X, self.W, self.H, self.C, self.mu, self.b1, self.b2,
                 self.lamW, self.lamH, self.optimizer, self.tol, self.max_iter,
                 partial=True)

        return self


    def apply_transform(self):
        X_new = model(self.W, self.H, self.mu, self.b1, self.b2)
        X_new = np.clip(X_new, a_min = 0.0, a_max = 1.0)
        parameters = {"W": self.W,
                      "H": self.H,
                      "mu": self.mu,
                      "b1": self.b1,
                      "b2": self.b2}
        return X_new, parameters


class CFStacker(BaseEstimator):

    def __init__(self,
                  base_estimator,
                  latent_dim,
                  matrix_factorization = True,
                  method='mean',
                  lamW=0.0,
                  lamH=0.0,
                  tol = 0.0001,
                  max_iter = 500,
                  learning_rate=0.1
                  ):

        self.base_estimator = base_estimator
        self.latent_dim = latent_dim
        self.matrix_factorization = matrix_factorization
        self.method = method
        self.lamW = lamW
        self.lamH = lamH
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        self.basemodel = self._generate_basemodel()
        if self.method == 'lr':
            self.output_model = LogisticRegression()

        self.X_comb_masked = None
        self.X_train_shape = None
        self.X_comb_reestimated = None
        self.H = None
        self.W_train = None
        self.C_labels = None
        self.C_train = None

        self.C_predict = None
        self.W = None
        self.nmf_predict = None


    def fit(self, X, y):

        self.X_train = X

        # confidence is based on distance from label
        self.C_labels = 1 - np.abs(X - np.expand_dims(y, axis=1))

        self.basemodel.fit(X, self.C_labels) # fit confidence model

        self.C_train = self.C_labels #np.clip(self.basemodel.predict(X), a_min = 0, a_max = 1)

        # self.mf_model = MatrixFactorization(latent_dim=self.latent_dim,
        #                                     C=self.C_train,
        #                                     lamW=self.lamW,
        #                                     lamH=self.lamH,
        #                                     tol=self.tol,
        #                                     max_iter=self.max_iter,
        #                                     learning_rate=self.learning_rate)
        #
        # self.mf_model.fit_transform(X)
        #
        # self.X_train_new, self.training_params = self.mf_model.apply_transform()

        return self


    def predict(self, X):

        self.C_test = np.clip(self.basemodel.predict(X), a_min=0, a_max=1)

        self.C_comb = np.concatenate((self.C_train, self.C_test), axis=0)

        self.C_comb = self.C_comb

        self.X_comb = np.concatenate((self.X_train, X), axis=0)

        self.mf_model = MatrixFactorization(latent_dim=self.latent_dim,
                                            C=self.C_comb,
                                            lamW=self.lamW,
                                            lamH=self.lamH,
                                            tol=self.tol,
                                            max_iter=self.max_iter,
                                            learning_rate=self.learning_rate)

        self.mf_model.fit_transform(self.X_comb)

        self.X_comb_new, self.comb_params = self.mf_model.apply_transform()

        self.X_predict = self.X_comb_new[self.X_train.shape[0]::, :]

        if self.method == 'mean':
            self.X_predict = np.mean(self.X_predict, axis=1)
        elif self.method == 'median':
            self.X_predict = np.median(self.X_predict, axis=1)

        return self.X_predict


    def _generate_basemodel(self):
        return MultiOutputRegressor(estimator=self.base_estimator)


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
    # mf_model = MatrixFactorization(latent_dim = 8,
    #                                 max_iter=5000,
    #                                 learning_rate=0.05)

    # initializer = keras.initializers.RandomUniform(minval=0,
    #                                                   maxval=1, 
    #                                                   seed=None)

    # mf_model.fit_transform(X)

    # H = tf.Variable(initializer(shape = [8, np.shape(X)[1]],
    #                 dtype = tf.dtypes.float32),
    #                 trainable = True)

    # mf_model.partial_fit_transform(X, H)

    # X_new, parameters = mf_model.apply_transform()

    cf_stacker = CFStacker(LinearRegression(),
                           latent_dim=10,
                           max_iter=500,
                           tol=0.005,
                           lamW=0.05,
                           lamH=0.05)

    cf_stacker.fit(X_train, labels_train)

    X_predict_probs = cf_stacker.predict(X_test)

    X_predict = X_predict_probs
    X_predict[X_predict >= 0.5] = 1
    X_predict[X_predict < 0.5] = 0

    X_predict_mean = np.mean(X_test, axis=1)
    X_predict_mean[X_predict_mean >= 0.5] = 1
    X_predict_mean[X_predict_mean < 0.5] = 0

    ascore_base = sklearn.metrics.accuracy_score(labels_test, X_predict_mean)

    ascore = sklearn.metrics.accuracy_score(labels_test, X_predict)

    print(ascore_base)
    print(ascore)
