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


def fmax_score(y_true, y_pred, beta=1, display=False):
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


def wmse(X_true, X_pred, C=1):
    C = tf.constant(C, dtype=tf.dtypes.float32)
    se = tf.math.multiply(C, tf.pow(X_true - X_pred, 2))
    non_zero = tf.cast(se != 0, dtype=tf.dtypes.float32)
    return tf.reduce_sum(se) / tf.reduce_sum(non_zero)


def l2_reg(U, lam):
    return lam * (tf.reduce_mean(tf.pow(U, 2)))
    # return lam * (tf.reduce_mean(tf.math.abs(U)))


def model(W, H, mu, bw, bh):
    return tf.linalg.matmul(W, H) + mu + bw + bh


def logistic_regression(X, omega, beta):
    return tf.nn.sigmoid(tf.matmul(X, omega) + beta)


def format_lr(yh):
    return yh  # tf.expand_dims(yh[:, 1], axis=-1)


def bce_loss(y_true, y_pred):
    one_minus_y_pred = 1 - y_pred
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1)
    one_minus_y_pred = tf.clip_by_value(one_minus_y_pred, 1e-9, 1)
    
    neg_pos_ratio = np.count_nonzero(y_true) / np.count_nonzero(y_true) 

    bce = -tf.reduce_mean(neg_pos_ratio * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(one_minus_y_pred)) / neg_pos_ratio

    return bce


def define_variables(X_shape, latent_dim):
    initializer1 = keras.initializers.RandomUniform(minval=-0.01,
                                                   maxval=0.01,
                                                   seed=None)
    # initializer2 = keras.initializers.RandomUniform(minval=1.01,
    #                                                maxval=0.99,
    #                                                seed=None)
    X1, X2 = X_shape
    W = tf.Variable(initializer1(shape=[X1, latent_dim],
                                dtype=tf.dtypes.float32),
                    trainable=True)
    H = tf.Variable(initializer1(shape=[latent_dim, X2],
                                dtype=tf.dtypes.float32),
                    trainable=True)
    # omega = tf.Variable(tf.zeros([X_shape[1], 1]),
    #                     dtype=tf.dtypes.float32)
    # beta = tf.Variable(tf.convert_to_tensor(np.array([0.5], dtype=np.float32)), dtype=tf.dtypes.float32)
    omega = tf.Variable(tf.zeros([X_shape[1], 1]),
                        dtype=tf.dtypes.float32)
    beta = tf.Variable(tf.zeros([1]), dtype=tf.dtypes.float32)
    # omega = tf.Variable(initializer2(shape=[X_shape[1], 1],
    #                     dtype=tf.dtypes.float32),
    #                     trainable=True)
    # beta = tf.Variable(initializer2(shape=[1, 1],
    #                     dtype=tf.dtypes.float32),
    #                     trainable=True)
    return W, H, omega, beta


def calculate_biases(X):
    mu = np.mean(X)
    muw = np.expand_dims(np.mean(X, axis=1), axis=1)
    muh = np.expand_dims(np.mean(X, axis=0), axis=0)

    mu = tf.constant(mu, dtype=tf.dtypes.float32)
    bw = tf.constant(muw - mu, dtype=tf.dtypes.float32)
    bh = tf.constant(muh - mu, dtype=tf.dtypes.float32)
    return mu, bw, bh


def calc_C(X, y):  # return binary matrix
    # return tf.math.floor(1 - tf.math.abs(X - y) + 1 / 2)
    return 1 - tf.math.abs(X - y)


class MatrixFactorizationClassifier(BaseEstimator):

    def __init__(self,
                 latent_dim=10,
                 lam_WH=0.0,
                 lam_omega=0.0,
                 alpha=0.5,
                 tol=0.0001,
                 max_iter=500,
                 learning_rate=0.05,
                 method="mean"):
        self.latent_dim = latent_dim
        self.lam_WH = lam_WH
        self.lam_omega = lam_omega
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.method = method
        self.optimizer = keras.optimizers.Adam(self.learning_rate)

    def fit(self, X, y):

        print("TRAINING")

        self.X_train = tf.constant(X, dtype=tf.dtypes.float32)

        y = np.expand_dims(y, axis=-1)

        # self.y = tf.constant(np.expand_dims(y, axis=-1), dtype=tf.dtypes.int32)

        self.W_train, self.H, self.omega, self.beta = define_variables(np.shape(X), self.latent_dim)

        self.mu_train, self.bw_train, self.bh_train = calculate_biases(X)

        self.optimize_train(X_train=self.X_train, y=y)

        return self

    def predict(self, X):

        print("PREDICTING")

        self.X_test = tf.constant(X, dtype=tf.dtypes.float32)

        self.W_test, _, _, _ = define_variables(np.shape(X), self.latent_dim)

        self.mu_test, self.bw_test, self.bh_test = calculate_biases(X)

        self.optimize_test(X_train=self.X_train, X_test=self.X_test)

        self.y_predict = logistic_regression(self.Xh_test, self.omega, self.beta).numpy()[:, 0]

        return self.y_predict

    def train_losses(self, X, Xh, y, yh, W, H, omega):
        loss_mf = wmse(X, Xh, self.C_train) + l2_reg(W, self.lam_WH) + l2_reg(H, self.lam_WH)
        loss_lr = bce_loss(y, yh) + l2_reg(omega, self.lam_omega)
        combined_loss = self.alpha * loss_mf + (1 - self.alpha) * loss_lr
        return combined_loss, self.alpha * loss_mf, (1 - self.alpha) * loss_lr

    def test_loss(self, X, Xh, yh, W, H, C):
        return self.alpha * (wmse(X, Xh, C) + l2_reg(W, self.lam_WH) + l2_reg(H, self.lam_WH))


    def optimization_train_step(self, X_train, y):
        with tf.GradientTape() as tape:
            self.Xh_train = model(self.W_train, self.H, self.mu_train, self.bw_train, self.bh_train)
            self.yh_train = logistic_regression(self.Xh_train, self.omega, self.beta)
            self.C_train = calc_C(X_train, self.yh_train)
            combined_loss, mf_loss, lr_loss = self.train_losses(X_train, self.Xh_train, y, self.yh_train, self.W_train,
                                                                self.H, self.omega)

        gradients = tape.gradient(combined_loss, [self.W_train, self.H, self.omega, self.beta])

        self.optimizer.apply_gradients(zip(gradients, [self.W_train, self.H, self.omega, self.beta]))

        return combined_loss, mf_loss, lr_loss


    def optimization_test_step(self, X_train, X_test):
        with tf.GradientTape() as tape:
            self.Xh_test = model(self.W_test, self.H, self.mu_test, self.bw_test, self.bh_test)
            self.yh_test = logistic_regression(self.Xh_test, self.omega, self.beta)
            self.C_test = calc_C(X_test, self.yh_test)
            mf_loss = self.test_loss(X_train, self.Xh_train, self.yh_train, self.W_train, self.H, self.C_train) \
                      + self.test_loss(X_test, self.Xh_test, self.yh_test, self.W_test, self.H, self.C_test)

        gradients = tape.gradient(mf_loss, [self.W_test])

        self.optimizer.apply_gradients(zip(gradients, [self.W_test]))

        return mf_loss

    def optimize_train(self, X_train, y):
        step = 0
        self.Xh_train = model(self.W_train, self.H, self.mu_train, self.bw_train, self.bh_train)
        self.yh_train = format_lr(logistic_regression(self.Xh_train, self.omega, self.beta))
        self.C_train = calc_C(X_train, self.yh_train)
        combined_loss, mf_loss, lr_loss = self.train_losses(X_train, self.Xh_train, y, self.yh_train, self.W_train,
                                                            self.H, self.omega)

        while combined_loss > self.tol:

            combined_loss, mf_loss, lr_loss = self.optimization_train_step(X_train, y)

            step = step + 1

            if step % 100 == 0:
                print("epoch: %i, combined_loss: %f, mf_loss: %f, lr_loss: %f" % (step, combined_loss, mf_loss, lr_loss))

            if step == self.max_iter:
                print("Increase max_iter: unable to meet convergence criteria")
                break

    def optimize_test(self, X_train, X_test):
        step = 0
        self.Xh_test = model(self.W_test, self.H, self.mu_test, self.bw_test, self.bh_test)
        self.yh_test = format_lr(logistic_regression(self.Xh_test, self.omega, self.beta))
        self.C_test = calc_C(X_test, self.yh_test)
        mf_loss = self.test_loss(X_train, self.Xh_train, self.yh_train, self.W_train, self.H, self.C_train) \
                  + self.test_loss(X_test, self.Xh_test, self.yh_test, self.W_test, self.H, self.C_test)

        while mf_loss > self.tol:

            mf_loss = self.optimization_test_step(X_train, X_test)

            step = step + 1

            if step % 100 == 0:
                print("epoch: %i, mf_loss: %f" % (step, mf_loss))

            if step == 200:
                print("Increase max_iter: unable to meet convergence criteria")
                break


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
                                             alpha=0.95,
                                             max_iter=3000,
                                             learning_rate=0.001,
                                             tol=0.0000000001,
                                             lam_WH=0.0,
                                             lam_omega=0.0)
    mf_model.fit(X_train, y_train)
    # %%
    y_pred = mf_model.predict(X_test)

    sk_lr = LogisticRegression()
    sk_lr.fit(X_test, y_test)
    sk_y_pred = sk_lr.predict_proba(X_test)[:, 1]

    mean_y_pred = np.mean(X_test, axis=1)

    fmax_score(y_test, y_pred, display=True)
    fmax_score(y_test, mean_y_pred, display=True)
    fmax_score(y_test, sk_y_pred, display=True)
