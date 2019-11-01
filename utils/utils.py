import argparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Binarizer, LabelBinarizer, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, FeatureUnion
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Binarizer, LabelBinarizer, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.datasets import fetch_mldata


def get_args():
    argParser = argparse.ArgumentParser(description=__doc__)
    argParser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argParser.parse_args()
    return args


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])

    def plot_learning_curves(model, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        train_errors, val_errors = [], []
        for m in range(1, len(X_train)):
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
            val_errors.append(mean_squared_error(y_val, y_val_predict))

    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0, 1])

    def plot_precision_vs_recall(precisions, recalls):
        plt.plot(recalls, precisions, "b-", linewidth=2)
        plt.xlabel("Recall", fontsize=16)
        plt.ylabel("Precision", fontsize=16)
        plt.axis([0, 1, 0, 1])


def split_train_test(data=None, test_ratio=0.8):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def fetch_batch(data, epoch, batch_index, batch_size):
    num_records, num_feats = data.shape
    n_batches = int(np.ceil(num_records/ batch_size))
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    data_train = split_train_test()
    X_batch = data[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch