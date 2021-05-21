#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.decomposition import NMF


def my_distance_matrix(A, B, squared=False):
    M = A.shape[0]
    N = B.shape[0]

    A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(M, 1))
    D_squared = A_dots + B_dots - 2 * A.dot(B.T)

    if (squared is False):
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared


def numerai_score(y_true, y_pred):
    rank_pred = y_pred.groupby(eras).apply(
        lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0, 1]


def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


train = pd.read_csv('numerai_training_data.csv')
test = pd.read_csv('numerai_tournament_data.csv')
alldata = pd.concat([train[train.columns[3:-1]], test[train.columns[3:-1]]])
S = my_distance_matrix(alldata.values.T, alldata.values.T, squared=False)
params = {}
params['gamma'] = 1.
params['degree'] = 3
params['coef0'] = 1
A = pairwise_kernels(S / S.max(), metric='rbf', filter_params=True, **params)
nmf = NMF(n_components=128, beta_loss='kullback-leibler',
          max_iter=10000, solver='mu')
nmf.fit(A)
nmf_train = np.matmul(train[train.columns[3:-1]].values, nmf.components_.T)
nmf_test = np.matmul(test[train.columns[3:-1]].values, nmf.components_.T)
nmf_test_validation = nmf_test[test.data_type == 'validation']
poly = PolynomialFeatures(degree=1)
o = np.linalg.lstsq(poly.fit_transform(nmf_train), train.target, rcond=-1)[0]
a = np.linalg.lstsq(poly.fit_transform(nmf_test_validation),
                    test[test.data_type == 'validation'].target, rcond=-1)[0]
preds = pd.Series(
    np.matmul(poly.fit_transform(nmf_train), o), index=train.index)
eras = train.era.str.slice(3).astype(int)
print(correlation_score(train.target, preds))
print(numerai_score(train.target, preds))
numerai_score(train.target, preds)
te_target = np.matmul(poly.fit_transform(nmf_test_validation), o)
te_data = test[test.data_type == 'validation'].copy().reset_index(drop=True)
eras = te_data.era.str.slice(3).astype(int)
preds = pd.Series(np.matmul(poly.fit_transform(
    nmf_test_validation), o), index=te_data.index)
print(correlation_score(te_data.target, preds))
print(numerai_score(te_data.target, preds))
eras = te_data.era.str.slice(3).astype(int)
preds = pd.Series(np.matmul(poly.fit_transform(
    nmf_test_validation), a), index=te_data.index)
print(correlation_score(te_data.target, preds))
print(numerai_score(te_data.target, preds))
sub = pd.read_csv('example_predictions.csv')

sub.prediction = np.matmul(poly.fit_transform(nmf_test), a)
sub.to_csv('superofficial.csv', index=False)

w = np.genfromtxt(
    'W_numer_full_relu_2.csv', delimiter=',')
q = np.maximum(0, w)
poly = PolynomialFeatures(degree=1)
nmf_train = np.matmul(train[train.columns[3:-1]].values, q)
nmf_test = np.matmul(test[train.columns[3:-1]].values, q)
nmf_test_validation = nmf_test[test.data_type == 'validation']
o = np.linalg.lstsq(poly.fit_transform(nmf_train), train.target, rcond=-1)[0]
preds = pd.Series(
    np.matmul(poly.fit_transform(nmf_train), o), index=train.index)
eras = train.era.str.slice(3).astype(int)
print(correlation_score(train.target, preds))
print(numerai_score(train.target, preds))
a = np.linalg.lstsq(poly.fit_transform(nmf_test_validation),
                    test[test.data_type == 'validation'].target, rcond=-1)[0]
te_data = test[test.data_type == 'validation'].copy().reset_index(drop=True)
eras = te_data.era.str.slice(3).astype(int)
preds = pd.Series(np.matmul(poly.fit_transform(
    nmf_test_validation), o), index=te_data.index)
print(correlation_score(te_data.target, preds))
print(numerai_score(te_data.target, preds))

sub.prediction = np.matmul(poly.fit_transform(nmf_test), a)
sub.to_csv('superofficialII.csv', index=False)
