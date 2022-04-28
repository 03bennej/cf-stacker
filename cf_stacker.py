import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nmf import NMF
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

e
def print_metrics(y_true, y_pred):
    print('f1_score: ', sklearn.metrics.f1_score(y_true, y_pred))
    print('accuracy: ', sklearn.metrics.accuracy_score(y_true, y_pred))
    print('precision: ', sklearn.metrics.precision_score(y_true, y_pred))
    print('recall: ', sklearn.metrics.recall_score(y_true, y_pred))
    print('AUC: ', sklearn.metrics.roc_auc_score(y_true, y_pred))


def remove_unreliable_entries(data,
                              unreliable_entries,
                              threshold=0.5,
                              use_probs=False,
                              target=np.nan):
    data_new = np.copy(data)
    data_new[unreliable_entries > threshold] = target
    # if use_probs:
    #     data_new[unreliable_entries > threshold] = target   
    # else:
    #     data_new[unreliable_entries == 1] = target
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


class cf_stacker(BaseEstimator):

    def __init__(self,
                 base_estimator,
                 latent_dimension,
                 threshold=0.5,
                 alpha_nmf=0.0,
                 max_iter_nmf=500,
                 use_probs=False,
                 nmf=True,
                 return_probs=False):

        self.base_estimator = base_estimator
        self.threshold = threshold
        self.latent_dimension = latent_dimension
        self.alpha_nmf = alpha_nmf
        self.max_iter_nmf = max_iter_nmf
        self.use_probs = use_probs
        self.nmf = nmf
        self.return_probs = return_probs

        self.basemodel = self._generate_basemodel()

    def fit(self, X, y):

        y = np.abs(X - np.expand_dims(y, axis=1))
        self.basemodel.fit(X, y)

        if self.use_probs:
            probs_list = self.basemodel.predict_proba(X)
            self.mask_train = list_to_matrix(probs_list)
        else:
            self.mask_train = self.basemodel.predict(X)

        self.X_train_masked = remove_unreliable_entries(X,
                                                        unreliable_entries=self.mask_train,
                                                        use_probs=self.use_probs,
                                                        threshold=self.threshold)

        if self.nmf:
            self.nmf_train = NMF(n_components=self.latent_dimension,
                                 max_iter=self.max_iter_nmf,
                                 init='custom',
                                 solver='mu',
                                 alpha=self.alpha_nmf,
                                 update_H=True)
            self.W_train = self.nmf_train.fit_transform(self.X_train_masked,
                                                        W=np.random.rand(X.shape[0], self.latent_dimension),
                                                        H=np.random.rand(self.latent_dimension, X.shape[1], ))
            self.H = self.nmf_train.components_
        return self

    def predict(self, X, method='mean'):

        if self.use_probs:
            probs_list = self.basemodel.predict_proba(X)
            self.mask_predict = list_to_matrix(probs_list)
        else:
            self.mask_predict = self.basemodel.predict(X)

        self.X_predict_masked = remove_unreliable_entries(X,
                                                          unreliable_entries=self.mask_predict,
                                                          use_probs=self.use_probs,
                                                          threshold=self.threshold)

        if self.nmf:
            self.nmf_predict = NMF(n_components=self.latent_dimension,
                                   max_iter=self.max_iter_nmf,
                                   init='custom',
                                   solver='mu',
                                   alpha=self.alpha_nmf,
                                   update_H=False)
            self.W_predict = self.nmf_predict.fit_transform(self.X_predict_masked,
                                                            W=np.random.rand(X.shape[0], self.latent_dimension),
                                                            H=self.H)

            if method == 'mean':
                X_predict = np.mean(self.W_predict @ self.H, axis=1)
            elif method == 'median':
                X_predict = np.median(self.W_predict @ self.H, axis=1)
        else:
            if method == 'mean':
                X_predict = np.nanmean(self.X_predict_masked, axis=1)
            elif method == 'median':
                X_predict = np.nanmedian(self.X_predict_masked, axis=1)
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

    model1 = cf_stacker(base_estimator=base_estimator1,
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
