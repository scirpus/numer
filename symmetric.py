#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def numerai_score(y_true, y_pred, eras):
    rank_pred = y_pred.groupby(eras).apply(
        lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0, 1]


def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def CreateSubmission(submissionFile, q, train, test):
    poly = PolynomialFeatures(degree=1)
    nmf_train = np.matmul(train[train.columns[3:-1]].values,q)
    nmf_test = np.matmul(test[train.columns[3:-1]].values,q)
    nmf_test_validation = nmf_test[test.data_type=='validation']
    o = np.linalg.lstsq(poly.fit_transform(nmf_train),train.target,rcond=-1)[0]
    preds = pd.Series(np.matmul(poly.fit_transform(nmf_train),o), index=train.index)
    eras = train.era.str.slice(3).astype(int)
    print('Train Correlation Score:',correlation_score(train.target, preds))
    print('Train Numer.ai Score:',numerai_score(train.target, preds, eras))
    a = np.linalg.lstsq(poly.fit_transform(nmf_test_validation),
                        test[test.data_type=='validation'].target,
                        rcond=-1)[0]
    te_data = test[test.data_type=='validation'].copy().reset_index(drop=True)
    eras = te_data.era.str.slice(3).astype(int)
    preds2 = pd.Series(np.matmul(poly.fit_transform(nmf_test_validation),a),
                       index=te_data.index)
    print('Validation Correlation Score:', correlation_score(te_data.target, preds2))
    print('Validation Numer.ai Score:', numerai_score(te_data.target, preds2, eras))
    sub = test[['id']].copy()
    sub['prediction'] = np.matmul(poly.fit_transform(nmf_test), a)
    sub.to_csv(submissionFile, index=False)


if __name__ == "__main__":
    train = pd.read_csv('numerai_training_data.csv')
    test = pd.read_csv('numerai_tournament_data.csv')
    print('Kullback Leibler Model')
    w = np.genfromtxt('./kullbackleibler.csv', delimiter=',')
    CreateSubmission('superofficial.csv', w, train, test)

    print('Frobenius Model')
    w = np.genfromtxt('./frobenius.csv', delimiter=',')
    CreateSubmission('superofficialII.csv', w, train, test)

