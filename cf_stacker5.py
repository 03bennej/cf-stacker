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


def model(W, H):
    return tf.linalg.matmul(W, H)


def logistic_regression(X, omega, b):
    return tf.nn.softmax(tf.matmul(X, omega) + b)


def format_lr(yh):
    return tf.expand_dims(yh[:, 1], axis=-1)


def bce_loss(y_true, y_pred):
    
    y_true = tf.one_hot(y_true, depth=2)

    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    
    bce = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))
    
    return bce


def lr_optimization_step(X, y, W, b, optimizer, lam):

    with tf.GradientTape() as g:

        pred = logistic_regression(X, W, b)

        loss = bce_loss(pred, y, W, b, lam)

    gradients = g.gradient(loss, [W, b])

    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    return loss


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
    omega = tf.Variable(tf.zeros([X_shape[1], 2]), name="weight")
    beta = tf.Variable(tf.zeros([2]), name="bias")
    return W, H, omega, beta


def calc_C(X, y): # return binary matrix
    # return tf.math.floor(1 - tf.math.abs(X - y) + 1/2)
    return 1 - tf.math.abs(X - y)


def train_losses(X, Xh, y, yh, W, H, omega, lam1, lam2, alpha):
    C = calc_C(X, yh)
    loss_mf = wmse(X, Xh, C) + l2_reg(W, lam1) + l2_reg(H, lam1)
    loss_lr = bce_loss(y, yh) + l2_reg(omega, lam2)
    combined_loss = alpha * loss_mf + (1-alpha) * loss_lr
    return combined_loss, loss_mf, loss_lr


def test_loss(X, Xh, yh, W, H, lam1, alpha):
    C = calc_C(X, yh)
    return alpha * (wmse(X, Xh, C) + l2_reg(W, lam1) + l2_reg(H, lam1))    


def optimization_train_step(X, y, W, H, omega, b, lam1, lam2, alpha, optimizer):
    with tf.GradientTape() as tape:
        Xh = model(W, H)
        yh = format_lr(logistic_regression(Xh, omega, b))
        combined_loss, mf_loss, lr_loss = train_losses(X, Xh, y, yh, W, H, omega, lam1, lam2, alpha) 

    gradients = tape.gradient(combined_loss, [W, H, omega, b])

    optimizer.apply_gradients(zip(gradients, [W, H, omega, b]))
    
    return combined_loss, mf_loss, lr_loss


def optimization_test_step(X_train, X_test, W_train, W_test, H, omega, b, lam1, lam2, alpha, optimizer):
    X_comb = tf.concat((X_train, X_test), axis=0)
    W_comb = tf.concat((W_train, W_test), axis=0)
    with tf.GradientTape() as tape:
        Xh = model(W_test, H)
        yh = format_lr(logistic_regression(Xh, omega, b))
        mf_loss = test_loss(X_test, Xh, yh, W_test, H, lam1, alpha)

    gradients = tape.gradient(mf_loss, [W_test])

    optimizer.apply_gradients(zip(gradients, [W_test]))
    
    return mf_loss


def optimize_train(X, y, W, H, omega, b, lam1, lam2, alpha, optimizer, tol, max_iter):
    step = 0
    Xh = model(W, H)
    yh = format_lr(logistic_regression(Xh, omega, b))
    combined_loss, mf_loss, lr_loss = train_losses(X, Xh, y, yh, W, H, omega, lam1, lam2, alpha) 

    while combined_loss > tol:

        combined_loss, mf_loss, lr_loss = optimization_train_step(X, y, W, H, omega, b, lam1, lam2, alpha, optimizer)

        step = step + 1

        if step % 100 == 0:

            print("epoch: %i, combined_loss: %f, mf_loss: %f, lr_loss: %f" % (step, combined_loss, mf_loss, lr_loss))

        if step == max_iter:
            print("Increase max_iter: unable to meet convergence criteria")
            break
        
def optimize_test(X_train, X_test, W_train, W_test, H, omega, b, lam1, lam2, alpha, optimizer, tol, max_iter):
    step = 0
    X_comb = tf.concat((X_train, X_test), axis=0)
    W_comb = tf.concat((W_train, W_test), axis=0)
    Xh = model(W_comb, H)
    yh = format_lr(logistic_regression(Xh, omega, b))
    mf_loss = test_loss(X_comb, Xh, yh, W_comb, H, lam1, alpha) 

    while mf_loss > tol:

        mf_loss = optimization_test_step(X_train, X_test, W_train, W_test, H, omega, b, lam1, lam2, alpha, optimizer)

        step = step + 1

        if step % 100 == 0:

            print("epoch: %i, mf_loss: %f" % (step, mf_loss))

        if step == max_iter:
            print("Increase max_iter: unable to meet convergence criteria")
            break


class MatrixFactorizationClassifier(BaseEstimator):

    def __init__(self,
                 latent_dim=50,
                 lam_WH=0.0,
                 lam_omega=0.0,
                 alpha=0.99,
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
        
        self.X_train = tf.constant(X, dtype=tf.dtypes.float32)
        
        y = tf.constant(np.expand_dims(y, axis=-1), dtype=tf.dtypes.int32)
        
        self.W_train, self.H, self.omega, self.beta = define_variables(np.shape(X), self.latent_dim)

        optimize_train(self.X_train, y, self.W_train, self.H, self.omega, self.beta, 
                       self.lam_WH, self.lam_omega, self.alpha, self.optimizer, 
                       self.tol, self.max_iter)

        return self
    
    
    def predict(self, X, max_iter):
        
        self.X_test = tf.constant(X, dtype=tf.dtypes.float32)
        
        self.W_test, _, _, _ = define_variables(np.shape(X), self.latent_dim)
        
        optimize_test(self.X_train, self.X_test, self.W_train, self.W_test, 
                      self.H, self.omega, self.beta, self.lam_WH, self.lam_omega, 
                      self.alpha, self.optimizer, self.tol, max_iter)
        
        self.y_predict =  logistic_regression(self.W_test @ self.H, self.omega, self.beta)   
            
        return self.y_predict
    
    
    def fit_lr(self, X, y):
        
        X_tf = tf.constant(X, dtype=tf.dtypes.float32)

        self.omega = tf.Variable(tf.zeros([X.shape[1], 2]), name="weight")
        self.beta = tf.Variable(tf.zeros([2]), name="bias")
        
        y_hat = logistic_regression(X_tf, self.omega, self.beta)
        loss = bce_loss(y, y_hat, self.omega, self.beta, lam=self.lam).numpy()
        step=0
        while loss > self.tol:
            loss = lr_optimization_step(X_tf, 
                                        y, 
                                        self.omega, 
                                        self.beta, 
                                        self.optimizer,
                                        lam=self.lam).numpy()
            step+=1
            # if step % 100 == 0:
            #     print(loss)
            if step == self.max_iter:
                print("Increase max_iter: unable to meet convergence criteria")
                break
        return self


    def predict_lr(self, X):
        self.X_pred_new = self.mf_transform(X)
        X_tf = tf.constant(self.X_pred_new, dtype=tf.dtypes.float32)
        self.y_pred = logistic_regression(X_tf, self.omega, self.beta).numpy()
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
                                             alpha=0.3,
                                             max_iter=2000,
                                             learning_rate=0.01,
                                             tol=0.0000000001,
                                             lam_WH=0.0,
                                             lam_omega=0.0)
    mf_model.fit(X_train, y_train)
    y_pred = mf_model.predict(X_train, max_iter=2000)[:, 1]
    
    sk_lr = LogisticRegression()
    sk_lr.fit(X_train, y_train)
    sk_y_pred = sk_lr.predict_proba(X_test)[:, 1]
    
    fmax_score(y_train, y_pred, display=True)
    fmax_score(y_test, sk_y_pred, display=True)

