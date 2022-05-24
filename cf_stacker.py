import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nmf import NMF
from sklearn.metrics import precision_recall_curve
import sklearn
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier


def print_metrics(y_true, y_pred):
    print('f1_score: ', sklearn.metrics.f1_score(y_true, y_pred))
    print('accuracy: ', sklearn.metrics.accuracy_score(y_true, y_pred))
    print('precision: ', sklearn.metrics.precision_score(y_true, y_pred))
    print('recall: ', sklearn.metrics.recall_score(y_true, y_pred))
    print('AUC: ', sklearn.metrics.roc_auc_score(y_true, y_pred))


def thresholds(X, y, beta=1):
    # beta = 0 for precision, beta -> infinity for recall, beta=1 for harmonic mean
    X_temp = np.copy(X)
    thresholds = []
    for i in range(np.shape(X_temp)[1]):
        precision, recall, threshold = precision_recall_curve(y, X_temp[:, i])
        fmeasure = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
        argmax = np.nanargmax(fmeasure)
        # if threshold[argmax] < 0.5:
        #     thres = 1-threshold[argmax]
        # else:
        #     thres = threshold[argmax]
        th = threshold[argmax] if argmax < len(threshold) else 1.0
        thresholds.append(th)
        # thresholds.append(thres)
    return thresholds


def generate_mask(unreliable_probs,
                  threshold=0.5):
    mask = unreliable_probs >= threshold
    # total_entries = len(np.matrix.flatten(mask))
    # total_unreliable_probs = len(np.matrix.flatten(mask[mask]))
    # print("Percentage removed", total_unreliable_probs/total_entries)
    return mask


def apply_mask(data, mask, target=np.nan):
    data_new = np.copy(data)
    data_new[mask] = target
    return data_new


def restore_reliable_probs(data_new,
                           data_old,
                           mask):
    data_new[np.invert(mask)] = data_old.to_numpy()[np.invert(mask)]
    return data_new

def list_to_matrix(probs_list):
    if type(probs_list) is list:
        list_len = len(probs_list)
        n_points = probs_list[0].shape[0]
        matrix = np.ones((n_points, list_len))
        for i in range(list_len):
            matrix[:, i] = probs_list[i][:, 1]
    else:
        matrix = probs_list
    return matrix


class CFStacker(BaseEstimator):

    def __init__(self,
                 base_estimator,
                 latent_dimension,
                 threshold=0.5,
                 alpha_nmf=0.0,
                 max_iter_nmf=500,
                 tol_nmf=1e-4,
                 l1_ratio_nmf=0.0,
                 nmf=True,
                 return_probs=False,
                 method='mean'):

        self.base_estimator = base_estimator
        self.threshold = threshold
        self.latent_dimension = latent_dimension
        self.alpha_nmf = alpha_nmf
        self.max_iter_nmf = max_iter_nmf
        self.tol_nmf = tol_nmf
        self.l1_ratio = l1_ratio_nmf
        self.nmf = nmf
        self.return_probs = return_probs
        self.method = method

        self.basemodel = self._generate_basemodel()
        if self.method == 'lr':
            self.output_model = LogisticRegression()

        self.X_comb_masked = None
        self.X_train_shape = None
        self.X_comb_reestimated = None
        self.H = None
        self.W_train = None
        self.nmf_train = None
        self.X_train_masked = None
        self.mask_train = None
        self.unreliable_probs_labels = None
        self.unreliable_probs_train = None

        self.X_predict_masked = None
        self.mask_predict = None
        self.unreliable_probs_predict = None
        self.W = None
        self.nmf_predict = None

    def fit(self, X, y):

        if self.threshold == 'variable':
            self.threshold = thresholds(X, y)  # calculate variable thresholds using fmeasure

        self.unreliable_probs_labels = np.abs(X - np.expand_dims(y, axis=1))  # define unreliable probabilities

        self.basemodel.fit(X, self.unreliable_probs_labels)  # fit model to learn unreliable probabilities

        self.unreliable_probs_train = self.basemodel.predict(X)

        self.mask_train = generate_mask(self.unreliable_probs_train,
                                        threshold=self.threshold)

        self.X_train_masked = apply_mask(data=X,
                                         mask=self.mask_train,
                                         target=np.nan)

        if self.nmf:
            self.X_train_shape = np.shape(X)

            self.nmf_train = NMF(n_components=self.latent_dimension,
                                 max_iter=self.max_iter_nmf,
                                 init='custom',
                                 solver='mu',
                                 tol=self.tol_nmf,
                                 l1_ratio=self.l1_ratio,
                                 alpha=self.alpha_nmf,
                                 update_H=True)

            W_init = np.random.rand(X.shape[0], self.latent_dimension)
            H_init = np.random.rand(self.latent_dimension, X.shape[1])

            self.W_train = self.nmf_train.fit_transform(self.X_train_masked,
                                                        W=W_init,
                                                        H=H_init)
            self.H = self.nmf_train.components_

        if self.method == 'lr':
            if self.nmf:
                X_temp = restore_reliable_probs(data_new=self.W_train @ self.H,
                                                data_old=X,
                                                mask=self.mask_train)
                self.output_model.fit(X_temp, y)
            else:
                self.output_model.fit(X, y)

        return self

    def predict(self, X):

        self.unreliable_probs_predict = self.basemodel.predict(X)

        self.mask_predict = generate_mask(self.unreliable_probs_predict,
                                          threshold=self.threshold)

        self.X_predict_masked = apply_mask(data=X,
                                           mask=self.mask_predict,
                                           target=np.nan)

        if self.nmf:

            self.X_comb_masked = np.concatenate((self.X_train_masked, self.X_predict_masked), axis=0)

            self.nmf_predict = NMF(n_components=self.latent_dimension,
                                   max_iter=self.max_iter_nmf,
                                   init='custom',
                                   solver='mu',
                                   tol=self.tol_nmf,
                                   l1_ratio=self.l1_ratio,
                                   alpha=self.alpha_nmf,
                                   update_H=True)

            W_init = np.concatenate((self.W_train,
                                     np.random.rand(X.shape[0], self.latent_dimension)),
                                    axis=0)

            self.W = self.nmf_predict.fit_transform(self.X_comb_masked,
                                                    W=W_init,
                                                    H=self.H)
            self.H = self.nmf_predict.components_
            self.X_comb_reestimated = self.W @ self.H

            X_predict = self.X_comb_reestimated[self.X_train_shape[0]::, :]

            # X_predict = restore_reliable_probs(data_new=X_predict,
            #                                    data_old=X,
            #                                    mask=self.mask_predict)

            if self.method == 'mean':
                X_predict = np.mean(X_predict, axis=1)
            elif self.method == 'median':
                X_predict = np.median(X_predict, axis=1)
            elif self.method == 'lr':
                X_predict = self.output_model.predict_proba(X_predict)[:, 1]
        else:
            if self.method == 'mean':
                X_predict = np.nanmean(self.X_predict_masked, axis=1)
            elif self.method == 'median':
                X_predict = np.nanmedian(self.X_predict_masked, axis=1)
            elif self.method == 'lr':
                X_predict = self.output_model.predict_proba(X)[:, 1]
        if self.return_probs:
            return X_predict
        else:
            X_predict[X_predict > 0.5] = 1
            X_predict[X_predict <= 0.5] = 0
            return X_predict

    def _generate_basemodel(self):
        # return MultiOutputClassifier(estimator=self.base_estimator)
        return MultiOutputRegressor(estimator=self.base_estimator)


if __name__ == "__main__":
    train_data = pd.read_csv('validation-10000.csv.gz')
    test_data = pd.read_csv('predictions-10000.csv.gz')
    train_data = train_data.drop(["id"], axis=1)
    test_data = test_data.drop(["id"], axis=1)
    X_train = train_data.drop(["label"], axis=1).to_numpy()
    labels_train = train_data.pop("label").to_numpy()
    y_train = np.abs(X_train - np.expand_dims(labels_train, axis=1))
    X_test = test_data.drop(["label"], axis=1).to_numpy()
    labels_test = test_data.pop("label").to_numpy()
    y_test = np.abs(X_test - np.expand_dims(labels_test, axis=1))

    base_estimator1 = SVR(C=1, epsilon=0.1, max_iter=200)  # LinearRegression() #SVC(probability=True, C=1)
    base_estimator2 = MLPRegressor(hidden_layer_sizes=100,
                                   activation='logistic',
                                   learning_rate='adaptive',
                                   max_iter=200)

    model1 = CFStacker(base_estimator=base_estimator1,
                       latent_dimension=4,
                       threshold=0.5,
                       alpha_nmf=0.05,
                       max_iter_nmf=500,
                       use_probs=False,
                       nmf=True)
    model1.fit(X_train, y_train)

    predict_train_svm = model1.predict(X_train, return_prob=False, method='mean')
    predict_test_svm = model1.predict(X_test, return_prob=False, method='mean')

    model2 = cf_stacker(base_estimator=base_estimator1,
                        latent_dimension=5,
                        threshold=0.5,
                        alpha_nmf=0.1,
                        max_iter_nmf=500,
                        use_probs=False,
                        nmf=False)
    model2.fit(X_train, y_train)

    predict_train_without_nmf = model2.predict(X_train, return_prob=False, method='mean')
    predict_test_without_nmf = model2.predict(X_test, return_prob=False, method='mean')

    model3 = cf_stacker(base_estimator=base_estimator2,
                        latent_dimension=4,
                        threshold=0.5,
                        alpha_nmf=0.05,
                        max_iter_nmf=500,
                        use_probs=False,
                        nmf=True)
    model3.fit(X_train, y_train)

    predict_train_lr = model3.predict(X_train, return_prob=False, method='mean')
    predict_test_lr = model3.predict(X_test, return_prob=False, method='mean')

    predict_train_base = np.round(np.mean(X_train, axis=1)).astype(int)
    predict_test_base = np.round(np.mean(X_test, axis=1)).astype(int)

    print('\nSVM with NMF: \n')
    print_metrics(labels_test, predict_test_svm)

    print('\nSVM without NMF: \n')
    print_metrics(labels_test, predict_test_without_nmf)

    print('\nLinear Regression with NMF: \n')
    print_metrics(labels_test, predict_test_lr)

    print('\nSimple Average: \n')
    print_metrics(labels_test, predict_test_base)
