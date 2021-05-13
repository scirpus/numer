#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import MinMaxScaler


def _neutralize(df, columns, by, proportion=1.0):
    scores = df[columns]
    exposures = df[by].values
    scores = scores - proportion * \
        exposures.dot(np.linalg.pinv(exposures).dot(scores))
    return scores / scores.std()


def _normalize(df):
    X = (df.rank(method="first") - 0.5) / len(df)
    return sp.stats.norm.ppf(X)


def normalize_and_neutralize(df, columns, by, proportion=1.0):
    # Convert the scores to a normal distribution
    df[columns] = _normalize(df[columns])
    df[columns] = _neutralize(df, columns, by, proportion)
    return df[columns]


TOURNAMENT_NAME = ""
TARGET_NAME = f"target"
PREDICTION_NAME = f"prediction"
BENCHMARK = 0
BAND = 0.2


def score(df):
    # method="first" breaks ties based on order in array
    return np.corrcoef(
        df[TARGET_NAME],
        df[PREDICTION_NAME].rank(pct=True, method="first")
    )[0, 1]


# The payout function
def payout(scores):
    return ((scores - BENCHMARK) / BAND).clip(lower=-1, upper=1)


def Output(p):
    return (1. / (1 + np.exp(-p)))


def GPI(data):
    return Output(0.097947 * np.tanh((((data["feature_strength19"]) + (((((data["feature_charisma63"]) - (((data["feature_dexterity6"]) - (((data["feature_wisdom36"]) + (((((-((data["feature_constitution56"])))) + (((((data["feature_charisma75"]) - (data["feature_dexterity14"]))) - (data["feature_wisdom3"])))) / 2.0)))))))) - (((data["feature_dexterity14"]) - (((data["feature_charisma63"]) - (data["feature_dexterity7"])))))))) / 2.0)) +
                  0.099022 * np.tanh((((((((data["feature_charisma55"]) * (data["feature_strength34"]))) * (((data["feature_constitution42"]) * (data["feature_charisma76"]))))) + (((data["feature_strength1"]) - (((data["feature_constitution38"]) - (((((data["feature_charisma76"]) * (((data["feature_charisma55"]) * (data["feature_strength34"]))))) - ((((data["feature_constitution24"]) + (((data["feature_dexterity12"]) - (data["feature_charisma19"])))) / 2.0))))))))) / 2.0)) +
                  0.093451 * np.tanh(((((-((((((data["feature_constitution62"]) - (((data["feature_constitution18"]) + (((data["feature_charisma58"]) + ((((-((np.tanh((data["feature_constitution79"])))))) - (data["feature_constitution6"]))))))))) / 2.0))))) + ((((data["feature_wisdom20"]) + (((data["feature_wisdom26"]) * (((data["feature_charisma41"]) - (data["feature_charisma43"])))))) / 2.0))) / 2.0)) +
                  0.100000 * np.tanh(((data["feature_constitution66"]) * (((data["feature_dexterity2"]) * (((data["feature_constitution94"]) * ((((-((((((-((((((data["feature_constitution111"]) * (np.tanh((data["feature_dexterity4"]))))) * (((data["feature_dexterity2"]) * (((data["feature_strength30"]) + (data["feature_dexterity4"])))))))))) + (((data["feature_constitution16"]) - (data["feature_charisma46"])))) / 2.0))))) * 2.0)))))))) +
                  0.100000 * np.tanh(((data["feature_intelligence2"]) * ((((data["feature_dexterity14"]) + ((((-1.0)) + (((((((data["feature_wisdom44"]) + (data["feature_charisma10"])) / 2.0)) + (((((((data["feature_dexterity9"]) + (((data["feature_wisdom35"]) - (data["feature_charisma69"]))))) - (((data["feature_intelligence2"]) * (data["feature_intelligence2"]))))) - (data["feature_charisma69"])))) / 2.0))))) / 2.0)))) +
                  0.100000 * np.tanh((((((data["feature_intelligence6"]) + (((data["feature_wisdom23"]) * (((data["feature_wisdom23"]) - (data["feature_charisma10"])))))) / 2.0)) * ((((((data["feature_wisdom23"]) - (data["feature_charisma16"]))) + ((-((((data["feature_wisdom37"]) - ((((((((data["feature_charisma10"]) + (((data["feature_charisma3"]) - (data["feature_constitution50"]))))) * 2.0)) + (data["feature_wisdom23"])) / 2.0)))))))) / 2.0)))) +
                  0.099804 * np.tanh(((data["feature_charisma50"]) * ((-((((((((data["feature_charisma50"]) - (np.tanh(((((((data["feature_wisdom18"]) + (data["feature_wisdom18"])) / 2.0)) - (data["feature_wisdom41"]))))))) * (data["feature_wisdom8"]))) * ((-((((((data["feature_charisma28"]) - (np.tanh((((data["feature_wisdom45"]) - (data["feature_strength1"]))))))) * (data["feature_wisdom18"]))))))))))))) +
                  0.099804 * np.tanh(((((data["feature_dexterity11"]) * (np.tanh((((data["feature_dexterity11"]) * (np.tanh((((((data["feature_constitution2"]) - (((data["feature_dexterity11"]) - ((((((data["feature_constitution65"]) + (((data["feature_intelligence9"]) / 2.0))) / 2.0)) * 2.0)))))) / 2.0)))))))))) * ((-((np.tanh((data["feature_dexterity11"])))))))) +
                  0.099609 * np.tanh(((((data["feature_dexterity9"]) * (((data["feature_wisdom13"]) * (((data["feature_charisma85"]) * ((((((((data["feature_wisdom13"]) + (((data["feature_constitution50"]) - (data["feature_constitution91"])))) / 2.0)) * (((data["feature_wisdom13"]) * (((((data["feature_dexterity9"]) * (((((data["feature_charisma81"]) * (data["feature_wisdom13"]))) * 2.0)))) * 2.0)))))) * 2.0)))))))) / 2.0)) +
                  0.100000 * np.tanh((-((((data["feature_dexterity4"]) * ((((data["feature_wisdom2"]) + (((data["feature_wisdom36"]) * (((data["feature_wisdom36"]) * (((((data["feature_wisdom1"]) - (((data["feature_wisdom36"]) + (np.tanh(((-((data["feature_strength10"])))))))))) - (((((data["feature_dexterity4"]) + (np.tanh(((-((data["feature_strength10"])))))))) + (data["feature_constitution55"])))))))))) / 2.0))))))))


def GPII(data):
    return Output(0.099902 * np.tanh((((((data["feature_charisma37"]) - (data["feature_dexterity11"]))) + (((data["feature_constitution81"]) - (((data["feature_dexterity3"]) - (((((((((((data["feature_charisma37"]) - (data["feature_dexterity11"]))) - (((data["feature_charisma69"]) - (data["feature_charisma19"]))))) + (((data["feature_charisma6"]) - (data["feature_constitution81"])))) / 2.0)) + (((data["feature_charisma6"]) - (data["feature_charisma69"])))) / 2.0))))))) / 2.0)) +
                  0.097556 * np.tanh(((((((data["feature_strength4"]) + (((((data["feature_charisma63"]) * (((data["feature_charisma63"]) * (((data["feature_strength34"]) + (((((((((data["feature_strength34"]) + (data["feature_strength36"]))) + (data["feature_strength36"]))) * (data["feature_strength36"]))) * (data["feature_dexterity1"]))))))))) - (((((data["feature_dexterity7"]) / 2.0)) * 2.0)))))) - (data["feature_constitution38"]))) / 2.0)) +
                  0.099902 * np.tanh((((((((data["feature_wisdom35"]) - (np.tanh((((((data["feature_wisdom16"]) * 2.0)) + (data["feature_constitution110"]))))))) / 2.0)) + (((((((-((((data["feature_constitution110"]) - (((data["feature_charisma58"]) + (data["feature_wisdom42"])))))))) - (data["feature_dexterity12"]))) + ((((data["feature_wisdom8"]) + ((-((((data["feature_strength2"]) - (data["feature_wisdom32"]))))))) / 2.0))) / 2.0))) / 2.0)) +
                  0.095112 * np.tanh(((data["feature_strength19"]) * ((((((((data["feature_charisma85"]) * (data["feature_strength19"]))) + (((data["feature_wisdom23"]) - (data["feature_strength19"])))) / 2.0)) + ((((data["feature_charisma78"]) + ((((((data["feature_constitution31"]) * (data["feature_charisma45"]))) + (((((data["feature_constitution31"]) * (data["feature_strength19"]))) - (((data["feature_constitution11"]) - ((-((data["feature_strength19"]))))))))) / 2.0))) / 2.0)))))) +
                  0.089736 * np.tanh((-((((data["feature_wisdom2"]) * (((data["feature_constitution46"]) * ((((-((((data["feature_wisdom2"]) - (((((((data["feature_charisma57"]) * 2.0)) / 2.0)) - ((-((((data["feature_constitution27"]) + (data["feature_charisma34"]))))))))))))) / 2.0))))))))) +
                  0.099511 * np.tanh(((data["feature_charisma50"]) * (((((((data["feature_charisma45"]) + (data["feature_charisma5"]))) + (data["feature_intelligence5"]))) * (((data["feature_wisdom10"]) * (((((data["feature_wisdom42"]) * (data["feature_wisdom27"]))) * ((((data["feature_constitution90"]) + (((((data["feature_charisma79"]) + (data["feature_intelligence5"]))) * (data["feature_wisdom42"])))) / 2.0)))))))))) +
                  0.100000 * np.tanh(((((((data["feature_charisma35"]) - (np.tanh((data["feature_constitution19"]))))) - (np.tanh((data["feature_strength13"]))))) * ((((data["feature_dexterity13"]) + ((((data["feature_dexterity13"]) + ((-((((((data["feature_constitution15"]) * (((data["feature_charisma74"]) + (data["feature_strength18"]))))) + (data["feature_constitution19"]))))))) / 2.0))) / 2.0)))) +
                  0.100000 * np.tanh(((((data["feature_dexterity2"]) + (((data["feature_dexterity8"]) + (((data["feature_intelligence8"]) * (((((-3.0)) + ((-((data["feature_constitution40"]))))) / 2.0)))))))) * ((((data["feature_constitution108"]) + ((-(((((data["feature_wisdom24"]) + (np.tanh((((((data["feature_dexterity2"]) * (data["feature_dexterity2"]))) * (data["feature_wisdom33"])))))) / 2.0)))))) / 2.0)))) +
                  0.099804 * np.tanh(((data["feature_wisdom22"]) * ((-((((((data["feature_charisma69"]) * (((((data["feature_constitution91"]) / 2.0)) + (((((data["feature_charisma35"]) + (((((-((data["feature_charisma69"])))) + (((((((data["feature_dexterity12"]) - (data["feature_strength12"]))) / 2.0)) * 2.0))) / 2.0)))) - (data["feature_constitution91"]))))))) / 2.0))))))) +
                  0.100000 * np.tanh(((((data["feature_constitution108"]) * (((((-((data["feature_constitution6"])))) + (((((data["feature_wisdom11"]) * 2.0)) + (((np.tanh((((data["feature_constitution108"]) - (data["feature_charisma11"]))))) * 2.0))))) / 2.0)))) * (((((data["feature_wisdom1"]) * (((((data["feature_constitution96"]) - (data["feature_charisma11"]))) + (data["feature_wisdom39"]))))) * ((-((data["feature_constitution6"])))))))))


def GPIII(data):
    return Output(0.099902 * np.tanh((((((((data["feature_charisma76"]) * (data["feature_strength1"]))) + (((data["feature_charisma19"]) - (data["feature_dexterity11"]))))) + (((((((-((data["feature_dexterity7"])))) + ((((((data["feature_charisma54"]) - (data["feature_dexterity7"]))) + (((((((data["feature_charisma76"]) * (data["feature_charisma67"]))) * 2.0)) * (data["feature_dexterity7"])))) / 2.0))) / 2.0)) * 2.0))) / 2.0)) +
                  0.100000 * np.tanh(((((((((((((((np.tanh((data["feature_charisma28"]))) - (data["feature_constitution114"]))) + (((((data["feature_charisma5"]) * (data["feature_charisma5"]))) / 2.0)))) + (((data["feature_wisdom35"]) - ((((data["feature_dexterity12"]) + (data["feature_constitution102"])) / 2.0)))))) / 2.0)) - ((((data["feature_dexterity12"]) + (data["feature_constitution102"])) / 2.0)))) + (data["feature_charisma63"]))) / 2.0)) +
                  0.100000 * np.tanh((((((data["feature_strength34"]) + (np.tanh((np.tanh(((-((((((((data["feature_dexterity3"]) * 2.0)) * 2.0)) - (((data["feature_charisma77"]) - (((data["feature_constitution84"]) - (((data["feature_constitution34"]) - (((data["feature_wisdom16"]) - (((data["feature_constitution34"]) - ((((data["feature_charisma83"]) + (((data["feature_dexterity4"]) * 2.0))) / 2.0)))))))))))))))))))))) / 2.0)) / 2.0)) +
                  0.100000 * np.tanh(((data["feature_charisma85"]) * (((data["feature_strength19"]) * (((data["feature_strength19"]) * (((((data["feature_constitution42"]) * (data["feature_dexterity9"]))) * (((data["feature_strength19"]) * (((((((data["feature_strength1"]) * (((data["feature_strength19"]) + (data["feature_charisma2"]))))) + ((-((((data["feature_constitution42"]) - (data["feature_dexterity9"])))))))) + (data["feature_charisma2"]))))))))))))) +
                  0.100000 * np.tanh(np.tanh(((((((((((data["feature_constitution101"]) / 2.0)) / 2.0)) + (((data["feature_wisdom10"]) * (np.tanh((((np.tanh(((((((-((data["feature_strength9"])))) * 2.0)) * 2.0)))) * 2.0))))))) / 2.0)) * (((data["feature_dexterity6"]) - (((data["feature_wisdom42"]) + ((((data["feature_strength10"]) + (((data["feature_wisdom42"]) - (data["feature_dexterity6"])))) / 2.0)))))))))) +
                  0.100000 * np.tanh(((((0.0)) + ((((data["feature_wisdom23"]) + ((-(((((((data["feature_constitution7"]) + ((-((((data["feature_charisma10"]) - (data["feature_intelligence4"]))))))) / 2.0)) + ((((data["feature_charisma69"]) + ((((data["feature_constitution47"]) + (np.tanh((((data["feature_constitution46"]) - (((data["feature_charisma10"]) - (data["feature_constitution47"])))))))) / 2.0))) / 2.0)))))))) / 2.0))) / 2.0)) +
                  0.100000 * np.tanh(((data["feature_dexterity1"]) * ((((-((((data["feature_charisma6"]) * (((data["feature_wisdom43"]) - (((((((-(((-((((data["feature_strength3"]) + (data["feature_charisma45"]))))))))) * (((data["feature_charisma45"]) * (((data["feature_constitution70"]) * 2.0)))))) + (((((data["feature_strength3"]) * 2.0)) / 2.0))) / 2.0))))))))) / 2.0)))) +
                  0.094135 * np.tanh((((-(((((-((data["feature_wisdom4"])))) * ((((data["feature_charisma29"]) + (((data["feature_strength24"]) * (((data["feature_strength15"]) + (((((data["feature_constitution63"]) + (data["feature_constitution54"]))) * 2.0))))))) / 2.0))))))) * (((data["feature_constitution85"]) * (((data["feature_constitution4"]) * ((-((((((data["feature_strength15"]) * (data["feature_constitution46"]))) * 2.0))))))))))) +
                  0.097165 * np.tanh((-(((((((((((data["feature_charisma13"]) + (((data["feature_wisdom26"]) + (data["feature_charisma79"]))))) * (data["feature_wisdom26"]))) + (data["feature_intelligence3"])) / 2.0)) * ((((((data["feature_intelligence3"]) + (((data["feature_wisdom42"]) * ((-((((data["feature_charisma13"]) + (((data["feature_charisma45"]) + (data["feature_charisma79"]))))))))))) / 2.0)) / 2.0))))))) +
                  0.097165 * np.tanh(((data["feature_constitution108"]) * (np.tanh((((data["feature_constitution108"]) * ((((data["feature_constitution39"]) + (((data["feature_dexterity8"]) - (((data["feature_intelligence8"]) + (((data["feature_constitution63"]) - (((data["feature_dexterity8"]) * ((((data["feature_dexterity5"]) + (((data["feature_dexterity2"]) - (((data["feature_constitution108"]) + (data["feature_constitution39"])))))) / 2.0))))))))))) / 2.0)))))))))


def GPIV(data):
    return Output(0.099902 * np.tanh(((data["feature_charisma63"]) - ((((((data["feature_dexterity6"]) + (data["feature_dexterity14"])) / 2.0)) + ((((((((-2.0)) * (((data["feature_wisdom23"]) - (((data["feature_charisma9"]) - ((-((((data["feature_dexterity4"]) - (((data["feature_strength34"]) - (((data["feature_dexterity6"]) - ((((data["feature_charisma58"]) + (data["feature_dexterity6"])) / 2.0))))))))))))))))) / 2.0)) / 2.0)))))) +
                  0.100000 * np.tanh((((((((data["feature_charisma13"]) * (data["feature_charisma54"]))) + ((-((((data["feature_constitution32"]) + (((data["feature_constitution91"]) + ((-((((((data["feature_dexterity1"]) * (data["feature_charisma37"]))) + (((data["feature_strength4"]) * (((data["feature_charisma37"]) + (((data["feature_constitution59"]) * (((data["feature_charisma54"]) + (data["feature_constitution18"])))))))))))))))))))))) / 2.0)) / 2.0)) +
                  0.094330 * np.tanh(((data["feature_wisdom7"]) * ((-((((data["feature_intelligence3"]) - ((((data["feature_strength22"]) + (((np.tanh((((data["feature_constitution104"]) * (data["feature_strength19"]))))) * (((((((data["feature_strength22"]) - ((-((data["feature_strength19"])))))) - ((-((data["feature_wisdom7"])))))) - ((-((data["feature_wisdom26"]))))))))) / 2.0))))))))) +
                  0.100000 * np.tanh(((data["feature_constitution101"]) * (((((((((data["feature_charisma53"]) + ((((((data["feature_charisma5"]) + (((((data["feature_constitution12"]) * ((((((data["feature_dexterity13"]) - (data["feature_constitution114"]))) + (data["feature_dexterity13"])) / 2.0)))) - (data["feature_constitution114"])))) / 2.0)) * 2.0)))) * (data["feature_wisdom42"]))) - (np.tanh((data["feature_wisdom16"]))))) / 2.0)))) +
                  0.093157 * np.tanh(((((((((((data["feature_dexterity4"]) * (data["feature_wisdom36"]))) * (data["feature_wisdom36"]))) + (np.tanh(((-3.0)))))) + (data["feature_charisma29"]))) * (np.tanh((((data["feature_dexterity7"]) + (np.tanh(((-((((((data["feature_wisdom36"]) - (((data["feature_constitution85"]) - (data["feature_strength30"]))))) * (data["feature_constitution78"])))))))))))))) +
                  0.099707 * np.tanh(((((data["feature_strength3"]) * ((((data["feature_charisma66"]) + (((((((((data["feature_intelligence11"]) + (data["feature_dexterity5"]))) * ((-((((data["feature_dexterity5"]) * ((-((data["feature_dexterity8"]))))))))))) - (data["feature_constitution4"]))) - (((data["feature_intelligence11"]) * (((data["feature_constitution16"]) * (data["feature_intelligence11"])))))))) / 2.0)))) * (data["feature_strength1"]))) +
                  0.099902 * np.tanh((((((data["feature_wisdom43"]) + ((((data["feature_constitution78"]) + (((data["feature_wisdom33"]) - (((data["feature_wisdom40"]) + (((data["feature_constitution108"]) + (data["feature_wisdom20"])))))))) / 2.0))) / 2.0)) * (((((-((((data["feature_charisma63"]) + (data["feature_dexterity11"])))))) + (((data["feature_wisdom20"]) * (((data["feature_wisdom33"]) * (((data["feature_wisdom33"]) * 2.0))))))) / 2.0)))) +
                  0.099609 * np.tanh(np.tanh((((((data["feature_charisma35"]) * ((-((((((data["feature_intelligence4"]) * ((((data["feature_intelligence4"]) + (((((data["feature_dexterity3"]) * (((data["feature_charisma35"]) * ((-((data["feature_charisma35"])))))))) * (((data["feature_dexterity3"]) * (((data["feature_intelligence5"]) * (((data["feature_charisma35"]) * 2.0))))))))) / 2.0)))) * 2.0))))))) / 2.0)))) +
                  0.099804 * np.tanh((((((data["feature_wisdom22"]) + (((data["feature_constitution16"]) * (((((np.tanh(((((data["feature_wisdom41"]) + (((data["feature_constitution97"]) - (((data["feature_constitution81"]) * (data["feature_constitution70"])))))) / 2.0)))) + ((-((((data["feature_constitution16"]) * (data["feature_dexterity12"])))))))) + ((-((data["feature_dexterity12"]))))))))) / 2.0)) * (((data["feature_constitution100"]) * (data["feature_constitution91"]))))) +
                  0.088270 * np.tanh(((data["feature_charisma75"]) * (((data["feature_strength30"]) * (((((np.tanh((((data["feature_intelligence12"]) + (((data["feature_intelligence12"]) - (((((data["feature_strength15"]) + (((((data["feature_constitution4"]) + (((data["feature_wisdom3"]) - (data["feature_constitution113"]))))) - (data["feature_constitution34"]))))) * 2.0)))))))) * (((data["feature_constitution113"]) * 2.0)))) / 2.0)))))))


def GPV(data):
    return Output(0.099902 * np.tanh((((data["feature_charisma18"]) + ((-(((((((data["feature_dexterity11"]) + (((((((((data["feature_dexterity14"]) + ((-((((data["feature_constitution81"]) - (((data["feature_charisma69"]) - (data["feature_strength22"]))))))))) / 2.0)) * 2.0)) + ((((data["feature_charisma69"]) + ((-((((data["feature_constitution89"]) - (((data["feature_dexterity14"]) - (data["feature_charisma42"]))))))))) / 2.0))) / 2.0))) / 2.0)) * 2.0)))))) / 2.0)) +
                  0.097556 * np.tanh(((((data["feature_dexterity7"]) * (((((data["feature_dexterity1"]) * (((((((data["feature_strength19"]) * (data["feature_charisma63"]))) * (data["feature_charisma63"]))) * 2.0)))) * (data["feature_strength4"]))))) - ((((data["feature_dexterity7"]) + (((((((((data["feature_dexterity7"]) - (data["feature_strength34"]))) * 2.0)) - (data["feature_charisma63"]))) / 2.0))) / 2.0)))) +
                  0.100000 * np.tanh((((data["feature_wisdom23"]) + ((-((((((((np.tanh((data["feature_intelligence2"]))) + ((-((((data["feature_constitution42"]) - ((((((data["feature_constitution110"]) + (np.tanh((data["feature_constitution62"])))) / 2.0)) * 2.0)))))))) / 2.0)) + ((((np.tanh((np.tanh((((((((data["feature_constitution62"]) * 2.0)) * 2.0)) * 2.0)))))) + (data["feature_intelligence8"])) / 2.0))) / 2.0)))))) / 2.0)) +
                  0.099902 * np.tanh(((data["feature_dexterity7"]) * ((((data["feature_wisdom42"]) + ((-(((((((data["feature_wisdom7"]) - ((-(((((((data["feature_dexterity7"]) + (data["feature_wisdom7"])) / 2.0)) - (data["feature_wisdom42"])))))))) + (((((data["feature_constitution78"]) - (data["feature_charisma53"]))) - (((data["feature_charisma85"]) - (data["feature_wisdom7"])))))) / 2.0)))))) / 2.0)))) +
                  0.099707 * np.tanh(((((data["feature_constitution85"]) * ((((((((data["feature_constitution50"]) + (data["feature_constitution38"])) / 2.0)) * (data["feature_constitution50"]))) * 2.0)))) * ((-((((((((data["feature_constitution38"]) + (data["feature_intelligence4"])) / 2.0)) + ((-((((data["feature_charisma55"]) - ((((((data["feature_constitution38"]) - (data["feature_dexterity12"]))) + (np.tanh((np.tanh((data["feature_intelligence4"])))))) / 2.0)))))))) / 2.0))))))) +
                  0.100000 * np.tanh((-(((-((((((((((((data["feature_wisdom26"]) + (((data["feature_charisma5"]) - (data["feature_strength32"])))) / 2.0)) * (data["feature_wisdom3"]))) + ((((((((((data["feature_wisdom35"]) * (data["feature_wisdom3"]))) + (((data["feature_charisma5"]) - (data["feature_wisdom25"])))) / 2.0)) * (data["feature_wisdom20"]))) - (np.tanh((data["feature_charisma16"])))))) / 2.0)) * (data["feature_wisdom3"]))))))))) +
                  0.098729 * np.tanh(((data["feature_dexterity4"]) * (((data["feature_dexterity4"]) * (((data["feature_dexterity4"]) * ((((((data["feature_charisma12"]) * (((((data["feature_constitution52"]) + (data["feature_wisdom11"]))) * (data["feature_charisma76"]))))) + (((data["feature_constitution18"]) - (np.tanh((((((data["feature_constitution98"]) + (data["feature_constitution16"]))) * 2.0))))))) / 2.0)))))))) +
                  0.099120 * np.tanh((((((data["feature_charisma10"]) / 2.0)) + ((((-((((((data["feature_constitution12"]) * (data["feature_constitution68"]))) - (((((data["feature_charisma13"]) - (((data["feature_constitution46"]) - (((data["feature_charisma6"]) - (((data["feature_intelligence3"]) * 2.0)))))))) / 2.0))))))) / 2.0))) / 2.0)) +
                  0.098925 * np.tanh(((data["feature_constitution54"]) * ((((-((((data["feature_strength15"]) - ((((data["feature_charisma3"]) + ((((((((((data["feature_charisma19"]) * (data["feature_strength15"]))) * 2.0)) * (((((data["feature_charisma19"]) * (data["feature_strength15"]))) * 2.0)))) + (((((data["feature_charisma19"]) * 2.0)) * (np.tanh((data["feature_wisdom19"])))))) / 2.0))) / 2.0))))))) / 2.0)))) +
                  0.099902 * np.tanh(((data["feature_charisma32"]) * (((data["feature_charisma58"]) * (((((data["feature_strength30"]) + (data["feature_wisdom29"]))) * (((data["feature_charisma77"]) * (((((data["feature_strength15"]) + (((((data["feature_wisdom29"]) + (data["feature_constitution30"]))) * (data["feature_strength1"]))))) * (((data["feature_intelligence7"]) * (data["feature_intelligence7"]))))))))))))))


def GPVI(data):
    return Output(0.100000 * np.tanh((((((data["feature_charisma37"]) * (((data["feature_charisma54"]) * (data["feature_charisma63"]))))) + (((((data["feature_charisma76"]) * (data["feature_strength34"]))) - (((((data["feature_dexterity14"]) - (((data["feature_charisma63"]) - (np.tanh((data["feature_dexterity6"]))))))) + (((data["feature_dexterity4"]) - (data["feature_constitution30"])))))))) / 2.0)) +
                  0.100000 * np.tanh((((data["feature_strength1"]) + (((((data["feature_wisdom23"]) - ((((data["feature_wisdom7"]) + (((data["feature_constitution114"]) - (np.tanh((np.tanh((data["feature_charisma10"])))))))) / 2.0)))) - (((data["feature_charisma69"]) + (((((data["feature_constitution75"]) - (((data["feature_wisdom36"]) / 2.0)))) / 2.0))))))) / 2.0)) +
                  0.100000 * np.tanh((((((((data["feature_dexterity13"]) * (((((data["feature_strength19"]) + ((-((np.tanh((np.tanh((data["feature_constitution110"])))))))))) * 2.0)))) + (((data["feature_charisma5"]) + ((-((np.tanh((((np.tanh((((((data["feature_dexterity13"]) * 2.0)) * (((data["feature_dexterity4"]) + (data["feature_charisma5"]))))))) + (data["feature_dexterity4"]))))))))))) / 2.0)) / 2.0)) +
                  0.099511 * np.tanh((((-((((((data["feature_constitution93"]) - (((((data["feature_charisma53"]) * ((-(((((data["feature_strength9"]) + (np.tanh((data["feature_wisdom22"])))) / 2.0))))))) * ((((-((((((((data["feature_constitution93"]) * (data["feature_wisdom35"]))) + (data["feature_wisdom35"]))) + (((data["feature_constitution93"]) * (data["feature_wisdom36"])))))))) * 2.0)))))) / 2.0))))) / 2.0)) +
                  0.096090 * np.tanh(((data["feature_dexterity4"]) * ((-((((data["feature_dexterity4"]) * (((((((data["feature_wisdom46"]) - (((((data["feature_wisdom42"]) * (data["feature_dexterity4"]))) - (np.tanh((((data["feature_wisdom46"]) - (((data["feature_constitution39"]) - (data["feature_wisdom46"]))))))))))) - (data["feature_wisdom42"]))) / 2.0))))))))) +
                  0.099902 * np.tanh(((data["feature_wisdom43"]) * (np.tanh((((((data["feature_charisma58"]) - (np.tanh((((((data["feature_intelligence2"]) + (((((data["feature_intelligence2"]) + (((data["feature_intelligence4"]) - (data["feature_charisma59"]))))) - (data["feature_constitution104"]))))) + (((data["feature_dexterity7"]) + (((data["feature_intelligence2"]) + (((data["feature_intelligence2"]) - (data["feature_constitution104"]))))))))))))) / 2.0)))))) +
                  0.100000 * np.tanh(((data["feature_dexterity14"]) * (((data["feature_charisma47"]) * ((((data["feature_dexterity9"]) + ((((((data["feature_constitution71"]) + (((((data["feature_wisdom10"]) * 2.0)) * (((((((data["feature_wisdom10"]) * 2.0)) - (((np.tanh((data["feature_constitution82"]))) * 2.0)))) * (data["feature_strength3"])))))) / 2.0)) - (((data["feature_strength13"]) * 2.0))))) / 2.0)))))) +
                  0.100000 * np.tanh(((((data["feature_wisdom10"]) * (((((data["feature_intelligence12"]) + (((((data["feature_dexterity6"]) * (((((data["feature_constitution63"]) - (((data["feature_charisma50"]) - (data["feature_constitution24"]))))) * ((-((((((data["feature_constitution63"]) - (((((data["feature_intelligence12"]) * (data["feature_wisdom8"]))) - (data["feature_constitution63"]))))) * 2.0))))))))) / 2.0)))) / 2.0)))) / 2.0)) +
                  0.100000 * np.tanh(((data["feature_constitution103"]) * (((data["feature_wisdom21"]) * (((data["feature_constitution2"]) * (((data["feature_constitution54"]) * (((data["feature_wisdom21"]) * (((data["feature_wisdom8"]) + (((((data["feature_constitution2"]) + (((((data["feature_constitution105"]) - (data["feature_wisdom13"]))) + (data["feature_intelligence9"]))))) * ((-((((data["feature_intelligence9"]) - (data["feature_wisdom13"])))))))))))))))))))) +
                  0.099804 * np.tanh(((((((data["feature_constitution86"]) * (data["feature_strength36"]))) * (data["feature_strength36"]))) * (((data["feature_strength36"]) * (((data["feature_strength36"]) * (np.tanh((((data["feature_charisma70"]) - ((((((((data["feature_constitution12"]) + (data["feature_constitution40"]))) + (np.tanh((data["feature_constitution6"])))) / 2.0)) + (((data["feature_wisdom38"]) - (((data["feature_charisma63"]) / 2.0)))))))))))))))))


def GPVII(data):
    return Output(0.099902 * np.tanh(((((((((((((data["feature_wisdom42"]) * (data["feature_wisdom44"]))) - (((data["feature_dexterity6"]) + ((-((((data["feature_charisma63"]) - ((-((((data["feature_constitution102"]) * (((data["feature_charisma81"]) - (((data["feature_dexterity12"]) * 2.0)))))))))))))))))) + (((data["feature_charisma81"]) * (data["feature_charisma63"])))) / 2.0)) * 2.0)) + ((-((data["feature_dexterity4"]))))) / 2.0)) +
                  0.100000 * np.tanh(((((data["feature_strength19"]) - (((data["feature_constitution85"]) + ((-(((((((((data["feature_strength1"]) * (data["feature_strength19"]))) * 2.0)) + (data["feature_charisma1"])) / 2.0))))))))) * ((((((((data["feature_strength1"]) * (data["feature_dexterity9"]))) * (data["feature_charisma85"]))) + (((data["feature_constitution81"]) * (data["feature_charisma54"])))) / 2.0)))) +
                  0.099902 * np.tanh((-(((((data["feature_charisma69"]) + (((((-((((((data["feature_charisma69"]) + (((data["feature_strength34"]) + (((data["feature_charisma6"]) - (data["feature_intelligence2"]))))))) + (((data["feature_charisma79"]) + (((data["feature_wisdom32"]) - (data["feature_intelligence2"])))))))))) + (data["feature_constitution7"])) / 2.0))) / 2.0))))) +
                  0.099511 * np.tanh(((((data["feature_strength19"]) * (((data["feature_constitution26"]) * (data["feature_constitution50"]))))) * (((data["feature_charisma46"]) - (((data["feature_constitution15"]) - (((data["feature_dexterity2"]) + (((((data["feature_constitution114"]) * (((data["feature_dexterity2"]) + (((data["feature_strength19"]) - (data["feature_constitution38"]))))))) - (data["feature_constitution114"]))))))))))) +
                  0.100000 * np.tanh(((((((-((data["feature_dexterity7"])))) + (((data["feature_wisdom20"]) - ((((((((data["feature_constitution46"]) - (((((data["feature_wisdom12"]) - (data["feature_strength1"]))) * ((-((data["feature_dexterity7"])))))))) - (data["feature_intelligence5"]))) + (((data["feature_wisdom12"]) - (data["feature_strength1"])))) / 2.0))))) / 2.0)) / 2.0)) +
                  0.099902 * np.tanh(((data["feature_charisma57"]) * ((-((((((((((data["feature_charisma83"]) * (data["feature_charisma57"]))) + ((-((data["feature_charisma36"]))))) / 2.0)) + (((((((((((data["feature_wisdom30"]) * (((data["feature_intelligence9"]) * 2.0)))) * (data["feature_charisma83"]))) * (data["feature_intelligence9"]))) * (data["feature_intelligence9"]))) * (data["feature_intelligence9"])))) / 2.0))))))) +
                  0.099511 * np.tanh(((data["feature_wisdom8"]) * ((-(((-((((((((((data["feature_constitution97"]) - (data["feature_charisma35"]))) + (data["feature_wisdom41"]))) * (data["feature_strength14"]))) * (((data["feature_strength9"]) * ((((((((data["feature_wisdom8"]) + (data["feature_constitution97"])) / 2.0)) + (data["feature_strength9"]))) * ((((data["feature_charisma28"]) + (data["feature_wisdom22"])) / 2.0)))))))))))))))) +
                  0.100000 * np.tanh((((-((((data["feature_constitution69"]) * (((((data["feature_strength10"]) * (((((((data["feature_charisma34"]) + (((((data["feature_constitution6"]) * (data["feature_constitution6"]))) - (data["feature_intelligence4"]))))) + (data["feature_constitution69"]))) * (data["feature_intelligence4"]))))) * (((((data["feature_constitution69"]) - (data["feature_dexterity9"]))) / 2.0))))))))) * 2.0)) +
                  0.100000 * np.tanh(np.tanh((((((data["feature_strength13"]) - (((((((((((((data["feature_charisma11"]) * (data["feature_dexterity7"]))) - (data["feature_strength13"]))) + (data["feature_charisma11"])) / 2.0)) - (data["feature_constitution65"]))) + (data["feature_dexterity11"])) / 2.0)))) * (((((((data["feature_charisma11"]) * (data["feature_charisma45"]))) - (data["feature_dexterity11"]))) / 2.0)))))) +
                  0.099316 * np.tanh(((((np.tanh((data["feature_wisdom42"]))) * (((((data["feature_wisdom19"]) * (((data["feature_constitution92"]) * 2.0)))) * (data["feature_wisdom42"]))))) * (((((data["feature_charisma50"]) * (((data["feature_charisma50"]) * ((((((data["feature_constitution102"]) + (((((data["feature_charisma37"]) * 2.0)) * 2.0)))) + (((data["feature_charisma37"]) + (data["feature_constitution102"])))) / 2.0)))))) / 2.0)))))


def GPVIII(data):
    return Output(0.099902 * np.tanh((((data["feature_charisma46"]) + ((-((((np.tanh((((data["feature_charisma69"]) * ((((((data["feature_dexterity7"]) + (((data["feature_dexterity7"]) - (data["feature_charisma11"]))))) + (((((data["feature_dexterity14"]) + (((data["feature_dexterity11"]) - (data["feature_wisdom5"]))))) + (data["feature_dexterity6"])))) / 2.0)))))) * 2.0)))))) / 2.0)) +
                  0.100000 * np.tanh((((((data["feature_constitution30"]) * (data["feature_charisma18"]))) + ((((((((((((data["feature_dexterity1"]) * 2.0)) * (((((data["feature_charisma63"]) * 2.0)) * (data["feature_strength1"]))))) + (data["feature_dexterity12"])) / 2.0)) * (((((((data["feature_strength3"]) * (data["feature_charisma63"]))) * ((((data["feature_charisma63"]) + (data["feature_strength14"])) / 2.0)))) * 2.0)))) - (data["feature_dexterity12"])))) / 2.0)) +
                  0.099902 * np.tanh((((((data["feature_dexterity3"]) * (((data["feature_wisdom36"]) - (((data["feature_wisdom12"]) - (((((data["feature_wisdom23"]) * (((((data["feature_charisma79"]) * 2.0)) / 2.0)))) - (data["feature_strength13"]))))))))) + (((((((((data["feature_strength14"]) / 2.0)) * 2.0)) - (((data["feature_constitution114"]) - (((data["feature_wisdom23"]) * (data["feature_charisma79"]))))))) / 2.0))) / 2.0)) +
                  0.100000 * np.tanh((((((np.tanh((((data["feature_strength9"]) * (data["feature_wisdom20"]))))) + ((-((((data["feature_constitution20"]) - (((data["feature_charisma28"]) * ((((((((((((((data["feature_charisma13"]) * (((data["feature_wisdom20"]) * 2.0)))) + (data["feature_strength21"])) / 2.0)) * 2.0)) * (data["feature_charisma13"]))) * (data["feature_intelligence6"]))) * 2.0)))))))))) / 2.0)) / 2.0)) +
                  0.100000 * np.tanh((((((data["feature_charisma85"]) + ((-((((data["feature_dexterity7"]) + (((((data["feature_intelligence4"]) + (((((((data["feature_dexterity6"]) - (data["feature_constitution39"]))) + (((data["feature_intelligence4"]) * ((((((data["feature_constitution58"]) + ((-((((data["feature_dexterity6"]) - (data["feature_constitution39"]))))))) / 2.0)) - (data["feature_charisma61"]))))))) * 2.0)))) / 2.0)))))))) / 2.0)) / 2.0)) +
                  0.099707 * np.tanh(((((data["feature_constitution41"]) * (((data["feature_charisma35"]) + (data["feature_constitution41"]))))) * (np.tanh((((data["feature_charisma35"]) * ((-(((((data["feature_constitution4"]) + ((-((((data["feature_dexterity2"]) * (((data["feature_constitution81"]) + ((((((data["feature_constitution81"]) + (data["feature_constitution20"])) / 2.0)) * (((data["feature_charisma67"]) + (data["feature_strength3"]))))))))))))) / 2.0))))))))))) +
                  0.096188 * np.tanh(((data["feature_wisdom24"]) * (((np.tanh((((((data["feature_constitution90"]) / 2.0)) - (np.tanh((((((((data["feature_intelligence2"]) - ((((data["feature_dexterity12"]) + (data["feature_intelligence2"])) / 2.0)))) + (((data["feature_intelligence2"]) + (np.tanh((np.tanh(((-((data["feature_strength13"])))))))))))) * 2.0)))))))) / 2.0)))) +
                  0.100000 * np.tanh(((data["feature_intelligence4"]) * (((data["feature_intelligence4"]) * ((-(((((((data["feature_intelligence11"]) * (data["feature_constitution75"]))) + (((((data["feature_constitution50"]) + (((data["feature_intelligence4"]) - (((((data["feature_constitution19"]) + (data["feature_intelligence4"]))) + (data["feature_charisma10"]))))))) / 2.0))) / 2.0))))))))) +
                  0.099316 * np.tanh((-(((((((((((((((data["feature_constitution64"]) * (data["feature_charisma6"]))) * (data["feature_wisdom7"]))) * 2.0)) + (data["feature_dexterity9"])) / 2.0)) * (((data["feature_wisdom7"]) - ((((data["feature_constitution113"]) + (((((data["feature_intelligence12"]) * ((((data["feature_constitution113"]) + (data["feature_constitution67"])) / 2.0)))) * 2.0))) / 2.0)))))) * (data["feature_charisma71"])))))) +
                  0.099902 * np.tanh(((((data["feature_dexterity7"]) * (((((((((data["feature_wisdom42"]) * (data["feature_wisdom22"]))) - (((data["feature_constitution78"]) + (((((data["feature_constitution56"]) + (((data["feature_constitution65"]) - (data["feature_charisma9"]))))) - (np.tanh((((data["feature_charisma82"]) * (data["feature_wisdom42"]))))))))))) + (data["feature_wisdom42"]))) / 2.0)))) / 2.0)))


def GPIX(data):
    return Output(0.099902 * np.tanh((((data["feature_charisma76"]) + ((-(((((data["feature_constitution110"]) + (((((((data["feature_dexterity11"]) + (((data["feature_dexterity4"]) - (data["feature_charisma85"]))))) * 2.0)) - (((((((((data["feature_charisma46"]) - (data["feature_dexterity14"]))) + (data["feature_constitution42"]))) - (data["feature_dexterity14"]))) + (data["feature_constitution42"])))))) / 2.0)))))) / 2.0)) +
                  0.100000 * np.tanh((((((data["feature_charisma28"]) + ((((((((data["feature_dexterity9"]) + (data["feature_charisma1"]))) * (((((((data["feature_charisma10"]) - (np.tanh((data["feature_charisma69"]))))) + (data["feature_strength34"]))) * 2.0)))) + (((data["feature_wisdom23"]) - ((((((((0.318310)) + (data["feature_charisma69"]))) + (data["feature_intelligence2"]))) * 2.0))))) / 2.0))) / 2.0)) / 2.0)) +
                  0.099902 * np.tanh(((data["feature_dexterity8"]) * (((data["feature_dexterity8"]) * (((data["feature_wisdom36"]) - (((data["feature_wisdom2"]) + ((((data["feature_constitution37"]) + ((-((((((data["feature_strength3"]) * (data["feature_strength1"]))) * 2.0)))))) / 2.0)))))))))) +
                  0.100000 * np.tanh(np.tanh(((((((data["feature_charisma79"]) + ((((((((data["feature_wisdom20"]) + (((data["feature_strength22"]) - ((((data["feature_wisdom1"]) + ((((((data["feature_constitution114"]) - (data["feature_charisma31"]))) + (((data["feature_constitution62"]) + (data["feature_strength22"])))) / 2.0))) / 2.0))))) / 2.0)) - ((((data["feature_charisma31"]) + ((((data["feature_constitution114"]) + (data["feature_intelligence4"])) / 2.0))) / 2.0)))) * 2.0))) / 2.0)) / 2.0)))) +
                  0.095112 * np.tanh((((data["feature_wisdom23"]) + (np.tanh((np.tanh((((((((((((data["feature_charisma36"]) + ((((-(((((data["feature_dexterity11"]) + (data["feature_intelligence4"])) / 2.0))))) * 2.0))) / 2.0)) + ((((-((((data["feature_dexterity2"]) * (data["feature_wisdom23"])))))) * 2.0))) / 2.0)) * 2.0)) * 2.0))))))) / 2.0)) +
                  0.100000 * np.tanh(((data["feature_dexterity11"]) * (((data["feature_dexterity5"]) * (((((((data["feature_dexterity4"]) * (data["feature_constitution55"]))) - (((data["feature_wisdom21"]) - (((data["feature_charisma63"]) * (((((data["feature_charisma63"]) * (data["feature_charisma13"]))) - (((data["feature_wisdom21"]) - (((data["feature_charisma63"]) * (data["feature_constitution41"]))))))))))))) / 2.0)))))) +
                  0.100000 * np.tanh((((((((((data["feature_strength1"]) * (((data["feature_wisdom8"]) * 2.0)))) + (((data["feature_charisma3"]) - (((data["feature_constitution12"]) + (((data["feature_strength13"]) - (((((((data["feature_charisma50"]) * (data["feature_wisdom37"]))) * (data["feature_wisdom37"]))) * (((((((data["feature_wisdom8"]) * 2.0)) * 2.0)) * (data["feature_charisma61"])))))))))))) / 2.0)) / 2.0)) / 2.0)) +
                  0.099804 * np.tanh(((((data["feature_charisma54"]) - (data["feature_constitution42"]))) * ((((((np.tanh(((((((-((((data["feature_charisma54"]) - (data["feature_constitution21"])))))) * (data["feature_constitution21"]))) * 2.0)))) + ((((data["feature_constitution80"]) + ((-((((((data["feature_charisma54"]) - (data["feature_constitution55"]))) - (((data["feature_constitution21"]) - (data["feature_constitution5"]))))))))) / 2.0))) / 2.0)) / 2.0)))) +
                  0.086217 * np.tanh(((data["feature_wisdom21"]) * (((data["feature_constitution88"]) * (((data["feature_constitution88"]) * (((((((((data["feature_constitution88"]) * (((data["feature_wisdom34"]) + ((((data["feature_wisdom34"]) + (data["feature_wisdom22"])) / 2.0)))))) * (data["feature_wisdom21"]))) - (((data["feature_constitution75"]) * (data["feature_charisma26"]))))) - (((data["feature_constitution88"]) * (data["feature_dexterity7"]))))))))))) +
                  0.099902 * np.tanh(((data["feature_charisma63"]) * ((((((((((data["feature_wisdom3"]) * ((-((data["feature_constitution6"])))))) * 2.0)) + (((((data["feature_constitution113"]) - (((data["feature_constitution58"]) - ((((((((data["feature_charisma63"]) * (np.tanh((data["feature_strength19"]))))) + (data["feature_charisma82"])) / 2.0)) * 2.0)))))) * (data["feature_dexterity1"])))) / 2.0)) / 2.0)))))


def GPX(data):
    return Output(0.097947 * np.tanh(((((data["feature_charisma19"]) - ((((data["feature_dexterity12"]) + (((((data["feature_dexterity7"]) - (data["feature_constitution42"]))) - (((((data["feature_charisma37"]) - (((data["feature_dexterity4"]) - ((((((data["feature_strength4"]) - (data["feature_charisma9"]))) + (((((data["feature_charisma67"]) - (data["feature_dexterity11"]))) - (data["feature_constitution110"])))) / 2.0)))))) * 2.0))))) / 2.0)))) / 2.0)) +
                  0.099804 * np.tanh(((data["feature_charisma85"]) * (((((((((data["feature_constitution97"]) + (data["feature_strength19"]))) * (((((data["feature_charisma46"]) + (data["feature_wisdom42"]))) / 2.0)))) * (((((data["feature_strength14"]) + (((data["feature_dexterity1"]) * (((data["feature_strength19"]) * (data["feature_constitution39"]))))))) * 2.0)))) * (((data["feature_strength19"]) * (((data["feature_charisma85"]) / 2.0)))))))) +
                  0.099902 * np.tanh((((data["feature_wisdom23"]) + (((((((((data["feature_charisma79"]) - (data["feature_dexterity14"]))) + ((-2.0)))) - ((((-(((((((((data["feature_constitution89"]) + (((data["feature_strength7"]) - (data["feature_wisdom23"]))))) * 2.0)) + (((((data["feature_wisdom34"]) * 2.0)) + (((np.tanh((data["feature_charisma86"]))) * 2.0))))) / 2.0))))) / 2.0)))) / 2.0))) / 2.0)) +
                  0.100000 * np.tanh(((((((data["feature_charisma63"]) * (((data["feature_charisma63"]) * (data["feature_dexterity9"]))))) + (np.tanh((((((-((data["feature_constitution114"])))) + ((((-((((data["feature_dexterity6"]) - (((((((data["feature_strength9"]) + ((((np.tanh(((-((data["feature_dexterity9"])))))) + (data["feature_charisma58"])) / 2.0))) / 2.0)) + (data["feature_charisma58"])) / 2.0))))))) * 2.0))) / 2.0)))))) / 2.0)) +
                  0.099902 * np.tanh((-((((((data["feature_constitution31"]) * (((((-1.0)) + ((((((((data["feature_dexterity10"]) + (data["feature_charisma81"])) / 2.0)) - (((data["feature_wisdom22"]) * (((((data["feature_wisdom5"]) - ((((data["feature_charisma81"]) + (data["feature_intelligence3"])) / 2.0)))) / 2.0)))))) * 2.0))) / 2.0)))) * (((data["feature_constitution8"]) - (((data["feature_dexterity6"]) - (data["feature_wisdom22"])))))))))) +
                  0.099804 * np.tanh(((((data["feature_constitution50"]) * ((((-((((data["feature_constitution50"]) * ((((data["feature_intelligence4"]) + (((((data["feature_wisdom7"]) + ((((-((((data["feature_wisdom42"]) * (((data["feature_wisdom42"]) * (data["feature_dexterity4"])))))))) - (((data["feature_wisdom42"]) * (data["feature_dexterity4"]))))))) - (data["feature_dexterity4"])))) / 2.0))))))) * 2.0)))) / 2.0)) +
                  0.100000 * np.tanh(((data["feature_wisdom8"]) * (((((data["feature_strength34"]) - (np.tanh(((((data["feature_constitution62"]) + ((((((data["feature_constitution53"]) + (((((data["feature_intelligence2"]) * 2.0)) + (((((data["feature_charisma19"]) + (((((data["feature_dexterity4"]) - (data["feature_constitution97"]))) * 2.0)))) * 2.0))))) / 2.0)) - (data["feature_strength3"])))) / 2.0)))))) / 2.0)))) +
                  0.099511 * np.tanh(((data["feature_strength28"]) * (((((data["feature_constitution44"]) / 2.0)) * (((data["feature_dexterity4"]) - ((((((((((data["feature_constitution7"]) + (((data["feature_constitution16"]) * (data["feature_strength28"]))))) * (((data["feature_constitution68"]) + (data["feature_constitution24"]))))) * (((data["feature_strength15"]) + (data["feature_constitution24"]))))) + (data["feature_constitution16"])) / 2.0)))))))) +
                  0.100000 * np.tanh(((data["feature_constitution113"]) * (((data["feature_charisma13"]) * (((((data["feature_intelligence5"]) * (((((data["feature_charisma47"]) - (data["feature_dexterity4"]))) + (((data["feature_charisma54"]) + (((((data["feature_charisma86"]) * (((((data["feature_wisdom35"]) * (data["feature_wisdom35"]))) * (data["feature_wisdom35"]))))) * 2.0)))))))) * (((data["feature_charisma77"]) * (data["feature_charisma13"]))))))))) +
                  0.100000 * np.tanh((((((data["feature_strength1"]) + ((((-((data["feature_dexterity7"])))) - (((((((((((((((data["feature_constitution46"]) * (data["feature_strength1"]))) * (((data["feature_constitution54"]) * 2.0)))) * (data["feature_charisma67"]))) * (((data["feature_constitution54"]) * 2.0)))) * (data["feature_constitution54"]))) * (data["feature_strength1"]))) * (((data["feature_strength1"]) * 2.0))))))) / 2.0)) / 2.0)))


def GPXI(data):
    return Output(0.100000 * np.tanh(((((((((data["feature_strength4"]) * (data["feature_charisma19"]))) + (((data["feature_strength34"]) + (((((((data["feature_charisma76"]) * (((data["feature_strength4"]) + (data["feature_wisdom8"]))))) * (((data["feature_charisma18"]) * (((data["feature_constitution42"]) + (data["feature_dexterity7"]))))))) - (((data["feature_dexterity7"]) * 2.0)))))))) / 2.0)) / 2.0)) +
                  0.099707 * np.tanh((-(((((data["feature_dexterity5"]) + ((((((data["feature_dexterity14"]) + ((((data["feature_constitution38"]) + (((((((data["feature_charisma85"]) * (data["feature_charisma69"]))) - (data["feature_charisma46"]))) - (((data["feature_dexterity5"]) * 2.0))))) / 2.0))) / 2.0)) - (((((data["feature_charisma85"]) * 2.0)) / 2.0))))) / 2.0))))) +
                  0.100000 * np.tanh((((-(((((((((((data["feature_constitution102"]) + (data["feature_constitution102"])) / 2.0)) + (((((np.tanh((((data["feature_constitution40"]) - (data["feature_charisma58"]))))) - (((data["feature_charisma10"]) - (data["feature_constitution10"]))))) - (data["feature_wisdom23"]))))) * ((((data["feature_strength19"]) + (data["feature_wisdom23"])) / 2.0)))) / 2.0))))) / 2.0)) +
                  0.099511 * np.tanh((-((((data["feature_wisdom3"]) * (((data["feature_constitution78"]) * (np.tanh((((data["feature_dexterity6"]) - (((data["feature_wisdom35"]) * ((((data["feature_charisma77"]) + (((((data["feature_intelligence5"]) + (data["feature_constitution78"]))) * (data["feature_constitution78"])))) / 2.0))))))))))))))) +
                  0.099902 * np.tanh((((((data["feature_constitution108"]) + (((data["feature_strength3"]) * (data["feature_constitution106"])))) / 2.0)) * ((((((data["feature_constitution52"]) + (data["feature_constitution50"])) / 2.0)) * (np.tanh((((data["feature_dexterity13"]) - ((((data["feature_constitution52"]) + (np.tanh((((((data["feature_constitution108"]) * (((data["feature_constitution82"]) * (data["feature_constitution108"]))))) / 2.0))))) / 2.0)))))))))) +
                  0.099511 * np.tanh(((((np.tanh((((((-((data["feature_constitution62"])))) + (((data["feature_wisdom10"]) * (((((((((data["feature_wisdom42"]) * (((((data["feature_charisma18"]) * 2.0)) + (data["feature_intelligence12"]))))) + (((data["feature_intelligence12"]) + (data["feature_constitution97"]))))) * (data["feature_wisdom42"]))) - (((data["feature_wisdom10"]) * (data["feature_constitution65"])))))))) / 2.0)))) / 2.0)) / 2.0)) +
                  0.100000 * np.tanh(((data["feature_charisma67"]) * (((((((((data["feature_wisdom36"]) / 2.0)) + (((data["feature_strength15"]) * ((-((((data["feature_wisdom46"]) - ((((-((((data["feature_wisdom46"]) - (data["feature_constitution9"])))))) * (((data["feature_constitution9"]) * (((data["feature_strength30"]) * (((data["feature_intelligence12"]) + (data["feature_intelligence12"])))))))))))))))))) / 2.0)) / 2.0)))) +
                  0.099707 * np.tanh(np.tanh((np.tanh((((np.tanh((((data["feature_charisma6"]) * (((data["feature_constitution33"]) * (((data["feature_charisma6"]) * (np.tanh((((data["feature_strength19"]) / 2.0)))))))))))) * (((data["feature_constitution1"]) + (((data["feature_constitution18"]) * (((data["feature_strength19"]) * (((data["feature_constitution18"]) * (((data["feature_constitution18"]) * (data["feature_strength19"]))))))))))))))))) +
                  0.094233 * np.tanh(((((data["feature_constitution31"]) * (((((((((data["feature_constitution31"]) - (((data["feature_charisma34"]) - (np.tanh(((-((((data["feature_intelligence2"]) * 2.0))))))))))) / 2.0)) / 2.0)) * (((data["feature_wisdom38"]) + (((((data["feature_intelligence2"]) + (data["feature_intelligence8"]))) + (((data["feature_wisdom38"]) - (data["feature_charisma34"]))))))))))) / 2.0)) +
                  0.090616 * np.tanh(((((((data["feature_wisdom9"]) * (((((((data["feature_charisma41"]) * (data["feature_wisdom41"]))) * ((((data["feature_strength7"]) + ((((data["feature_constitution90"]) + ((-((data["feature_intelligence3"]))))) / 2.0))) / 2.0)))) * 2.0)))) * ((((((data["feature_constitution90"]) + ((-((data["feature_intelligence3"]))))) / 2.0)) + (data["feature_wisdom42"]))))) * (data["feature_charisma41"]))))


def GP(data):
    return .1 * (GPI(data) +
                 GPII(data) +
                 GPIII(data) +
                 GPIV(data) +
                 GPV(data) +
                 GPVI(data) +
                 GPVII(data) +
                 GPVIII(data) +
                 GPIX(data) +
                 GPX(data))


tr = pd.read_csv('numerai_training_data.csv')
te = pd.read_csv('numerai_tournament_data.csv')

cols = tr.columns[3:]
cols

tr[PREDICTION_NAME] = GPI(tr)
train_correlations = tr.groupby("era").apply(score)
print(
    f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
print(
    f"On training the average per-era payout is {payout(train_correlations).mean()}")
print(f"Sharpe {train_correlations.mean()/train_correlations.std()}")

tr[PREDICTION_NAME] = GPII(tr)
train_correlations = tr.groupby("era").apply(score)
print(
    f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
print(
    f"On training the average per-era payout is {payout(train_correlations).mean()}")
print(f"Sharpe {train_correlations.mean()/train_correlations.std()}")


tr[PREDICTION_NAME] = GPIII(tr)
train_correlations = tr.groupby("era").apply(score)
print(
    f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
print(
    f"On training the average per-era payout is {payout(train_correlations).mean()}")
print(f"Sharpe {train_correlations.mean()/train_correlations.std()}")


tr[PREDICTION_NAME] = GPIV(tr)
train_correlations = tr.groupby("era").apply(score)
print(
    f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
print(
    f"On training the average per-era payout is {payout(train_correlations).mean()}")
print(f"Sharpe {train_correlations.mean()/train_correlations.std()}")


tr[PREDICTION_NAME] = GPV(tr)
train_correlations = tr.groupby("era").apply(score)
print(
    f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
print(
    f"On training the average per-era payout is {payout(train_correlations).mean()}")
print(f"Sharpe {train_correlations.mean()/train_correlations.std()}")


tr[PREDICTION_NAME] = GPVI(tr)
train_correlations = tr.groupby("era").apply(score)
print(
    f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
print(
    f"On training the average per-era payout is {payout(train_correlations).mean()}")
print(f"Sharpe {train_correlations.mean()/train_correlations.std()}")


tr[PREDICTION_NAME] = GPVII(tr)
train_correlations = tr.groupby("era").apply(score)
print(
    f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
print(
    f"On training the average per-era payout is {payout(train_correlations).mean()}")
print(f"Sharpe {train_correlations.mean()/train_correlations.std()}")


tr[PREDICTION_NAME] = GPVIII(tr)
train_correlations = tr.groupby("era").apply(score)
print(
    f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
print(
    f"On training the average per-era payout is {payout(train_correlations).mean()}")
print(f"Sharpe {train_correlations.mean()/train_correlations.std()}")


tr[PREDICTION_NAME] = GPIX(tr)
train_correlations = tr.groupby("era").apply(score)
print(
    f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
print(
    f"On training the average per-era payout is {payout(train_correlations).mean()}")
print(f"Sharpe {train_correlations.mean()/train_correlations.std()}")


tr[PREDICTION_NAME] = GPX(tr)
train_correlations = tr.groupby("era").apply(score)
print(
    f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
print(
    f"On training the average per-era payout is {payout(train_correlations).mean()}")
print(f"Sharpe {train_correlations.mean()/train_correlations.std()}")


validation_data = te[te.data_type == "validation"].copy()
validation_data[PREDICTION_NAME] = GP(validation_data)
validation_correlations = validation_data.groupby("era").apply(score)
print(
    f"On validation the correlation has mean {validation_correlations.mean()} and std {validation_correlations.std()}")
print(
    f"On validation the average per-era payout is {payout(validation_correlations).mean()}")
print(f"Sharpe {validation_correlations.mean()/validation_correlations.std()}")


ex = pd.read_csv('example_predictions.csv')
ex.prediction = GP(te)
ex.to_csv('standard.csv', index=False)


features = ['feature_dexterity4', 'feature_charisma63', 'feature_strength19',
            'feature_dexterity7', 'feature_wisdom42', 'feature_strength1',
            'feature_intelligence4', 'feature_dexterity11', 'feature_wisdom23',
            'feature_intelligence2', 'feature_charisma69',
            'feature_dexterity6', 'feature_dexterity12', 'feature_dexterity9',
            'feature_dexterity14', 'feature_strength34', 'feature_wisdom36',
            'feature_constitution114', 'feature_dexterity2',
            'feature_charisma35', 'feature_wisdom35', 'feature_charisma10',
            'feature_wisdom8', 'feature_charisma54', 'feature_charisma85',
            'feature_wisdom7', 'feature_charisma13', 'feature_wisdom22',
            'feature_charisma37', 'feature_constitution50',
            'feature_strength3', 'feature_charisma5', 'feature_charisma79',
            'feature_strength15', 'feature_wisdom20', 'feature_charisma19',
            'feature_constitution42', 'feature_charisma76',
            'feature_charisma50', 'feature_strength36', 'feature_dexterity1',
            'feature_strength13', 'feature_intelligence9',
            'feature_constitution16', 'feature_constitution81',
            'feature_constitution108', 'feature_charisma58',
            'feature_constitution6', 'feature_constitution110',
            'feature_wisdom21', 'feature_charisma6', 'feature_strength9',
            'feature_constitution113', 'feature_charisma45',
            'feature_constitution46', 'feature_constitution54',
            'feature_dexterity13', 'feature_dexterity8',
            'feature_constitution39', 'feature_charisma11',
            'feature_constitution38', 'feature_wisdom13', 'feature_wisdom3',
            'feature_charisma81', 'feature_intelligence12', 'feature_wisdom26',
            'feature_constitution62', 'feature_constitution63',
            'feature_charisma46', 'feature_intelligence3',
            'feature_strength22', 'feature_wisdom10', 'feature_intelligence5',
            'feature_constitution97', 'feature_constitution102',
            'feature_charisma28', 'feature_constitution91',
            'feature_dexterity3', 'feature_constitution88',
            'feature_charisma67', 'feature_dexterity5', 'feature_wisdom33',
            'feature_strength4', 'feature_strength10', 'feature_strength14',
            'feature_constitution4', 'feature_constitution12',
            'feature_constitution78', 'feature_intelligence11',
            'feature_constitution85', 'feature_strength30',
            'feature_constitution24', 'feature_wisdom2', 'feature_wisdom46',
            'feature_constitution18', 'feature_constitution7',
            'feature_wisdom34', 'feature_charisma83', 'feature_constitution34',
            'feature_charisma77', 'feature_wisdom43', 'feature_constitution21',
            'feature_constitution93', 'feature_charisma9',
            'feature_constitution41', 'feature_constitution69',
            'feature_charisma53', 'feature_wisdom12', 'feature_constitution30',
            'feature_constitution75', 'feature_constitution104',
            'feature_charisma55', 'feature_wisdom1', 'feature_constitution55',
            'feature_wisdom37', 'feature_charisma57', 'feature_wisdom41',
            'feature_charisma3', 'feature_constitution19',
            'feature_constitution2', 'feature_constitution31',
            'feature_constitution65', 'feature_wisdom16', 'feature_wisdom18',
            'feature_intelligence8', 'feature_charisma61',
            'feature_charisma75', 'feature_charisma18',
            'feature_constitution68', 'feature_constitution89',
            'feature_constitution90', 'feature_constitution58',
            'feature_constitution20', 'feature_wisdom5', 'feature_wisdom19',
            'feature_wisdom29', 'feature_intelligence7', 'feature_charisma36',
            'feature_wisdom32', 'feature_charisma47', 'feature_charisma1',
            'feature_charisma82', 'feature_charisma34',
            'feature_constitution56', 'feature_charisma86',
            'feature_constitution47', 'feature_constitution101',
            'feature_charisma2', 'feature_charisma31', 'feature_wisdom24',
            'feature_constitution70', 'feature_charisma29',
            'feature_constitution40', 'feature_constitution15',
            'feature_wisdom44', 'feature_intelligence6', 'feature_strength28',
            'feature_charisma16', 'feature_wisdom11', 'feature_constitution26',
            'feature_constitution8', 'feature_constitution53',
            'feature_dexterity10', 'feature_wisdom38', 'feature_charisma70',
            'feature_constitution37', 'feature_wisdom30', 'feature_strength7',
            'feature_wisdom45', 'feature_constitution92', 'feature_strength21',
            'feature_constitution5', 'feature_constitution80',
            'feature_constitution64', 'feature_constitution67',
            'feature_charisma71', 'feature_constitution86',
            'feature_charisma26', 'feature_charisma41',
            'feature_constitution105', 'feature_constitution103',
            'feature_wisdom27', 'feature_constitution59',
            'feature_constitution32', 'feature_charisma74',
            'feature_strength18', 'feature_constitution111',
            'feature_strength24', 'feature_wisdom4', 'feature_constitution94',
            'feature_constitution79', 'feature_strength12',
            'feature_constitution66', 'feature_constitution96',
            'feature_constitution84', 'feature_wisdom39',
            'feature_constitution27', 'feature_charisma66', 'feature_wisdom40',
            'feature_charisma43', 'feature_constitution82',
            'feature_constitution71', 'feature_charisma59',
            'feature_strength2', 'feature_charisma78', 'feature_charisma32',
            'feature_constitution98', 'feature_constitution11',
            'feature_constitution52', 'feature_charisma12', 'feature_wisdom25',
            'feature_strength32', 'feature_charisma42',
            'feature_constitution100', 'feature_constitution44']
validation_data = te.copy()

validation_data["preds"] = GP(validation_data)
validation_data["preds_neutralized"] = validation_data.groupby("era").apply(
    # neutralize by 50% within each era
    lambda x: normalize_and_neutralize(x, ["preds"], features, 0.5)
)
scaler = MinMaxScaler()
validation_data[PREDICTION_NAME] = scaler.fit_transform(
    validation_data[["preds_neutralized"]])  # transform back to 0-1
validation_correlations = validation_data.groupby("era").apply(score)
print(
    f"On validation the correlation has mean {validation_correlations.mean()} and std {validation_correlations.std()}")
print(
    f"On validation the average per-era payout is {payout(validation_correlations).mean()}")
print(f"Sharpe {validation_correlations.mean()/validation_correlations.std()}")


ex = pd.read_csv('example_predictions.csv')
ex.prediction = validation_data.prediction
ex.to_csv('weaksauce.csv', index=False, float_format='%.6f')
