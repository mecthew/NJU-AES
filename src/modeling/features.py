#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-11-16
import re
import string

import nltk as nltk
import numpy as np
from nltk.corpus import stopwords
from scipy.stats import chi
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.preprocessing import StandardScaler

from integration.common import get_logger
from modeling.constant import ROOT_DIR      
from modeling.constant import *
from integration.tools import *

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


# constants
WORDS_NUM = 'words_num'
SENT_NUM = 'sentences_num'
WORD_AVG_LEN = 'word_avg_len'
SENT_AVG_LEN = 'sent_avg_len'
WRONG_SPELL_WORDS_NUM = 'wrong_spell_words_num'

POS_NN = 'pos_nn'
POS_NNS = 'pos_nns'
POS_VB = 'pos_vb'
POS_VBD = 'pos_vbd'
POS_VBG = 'pos_vbg'
POS_VBN = 'pos_vbn'
POS_VBP = 'pos_vbp'
POS_VBZ = 'pos_vbz'
POS_UH = 'pos_uh'
POS_JJ = 'pos_jj'
POS_RP = 'pos_rp'
POS_RB = 'pos_rb'
POS_WDT = 'pos_wdt'
POS_PR = 'pos_pr'
POS_DT = 'pos_DT'
POS_CC = 'pos_CC'
POS_CD = 'pos_CD'

AT_SYMBOLS = ['CAPS',
              'PERSON',
              'NUM',
              'MONEY',
              'ORGANIZATION',
              'LOCATION',
              'DATE',
              'TIME',
              'STATE',
              'DR',
              'EMAIL',
              'CITY',
              'MONTH',
              'PERCENT'
              ]
AT_CAPS = 'at_caps'
AT_ORGANIZATION = 'at_organization'
AT_PERSON = 'at_person'
AT_LOCATION = 'at_location'
AT_MONEY = 'at_money'
AT_TIME = 'at_time'
AT_DATE = 'at_date'
AT_PERCENT = 'at_percent'
AT_STATE = 'at_state'
AT_NUM = 'at_num'
AT_DR = 'at_dr'
AT_EMAIL = 'at_email'
AT_CITY = 'at_city'
AT_MONTH = 'at_month'


def get_wrong_spell_words_num(words):
    with open(ROOT_DIR + '/resource/words.txt') as word_file:
        valid_words = set(word_file.read().split())
    result = []
    wrong_words = []

    for article in words:
        cnt = 0
        for word in article:
            if word not in valid_words and word.lower() not in valid_words and word not in AT_SYMBOLS:
                    # and word not in [',', '.', '?', '@', '!', '\'', ':', '-', '/', '\\']:
                wrong_words.append(word)
                cnt += 1
        result.append(cnt)
    wrong_words = set(wrong_words)

    return result


def replace_at_symbol(X):
    res = []

    for essay in X:
        for symbols in AT_SYMBOLS:
            essay = re.sub(f'@{symbols}[0-9]*', ' ' + symbols + ' ', essay)
        res.append(essay)

    res = np.array(res)
    return res


def get_all_features(X, remove_stopwords=False):
    # X = replace_at_symbol(X)

    words = list(map(lambda x: nltk.word_tokenize(re.sub('[%s]' % re.escape(string.punctuation), ' ', x)), X))
    sents = list(map(nltk.sent_tokenize, X))

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [[w for w in a if w not in stop_words] for a in words]
        words_without_stopwords = sum(map(len, words))
    pos_tags = list(map(nltk.pos_tag, words))

    features = {}

    # count
    features[WORDS_NUM] = list(map(len, words))
    features[SENT_NUM] = list(map(len, sents))
    features[WORD_AVG_LEN] = list(map(lambda x: len(''.join(x)) / len(x), words))
    features[SENT_AVG_LEN] = list(map(lambda x: len(words[x]) / len(sents[x]), range(len(words))))
    features[WRONG_SPELL_WORDS_NUM] = get_wrong_spell_words_num(words)

    # part of speech
    features[POS_NN] = [len(list(filter(lambda x: x[1][:2] == 'NN', a))) for a in pos_tags]
    # features[POS_NNS] = [len(list(filter(lambda x: x[1][:3] == 'NNS', a))) for a in pos_tags]
    features[POS_VB] = [len(list(filter(lambda x: x[1][:2] == 'VB', a))) for a in pos_tags]
    # features[POS_VBD] = [len(list(filter(lambda x: x[1][:3] == 'VBD', a))) for a in pos_tags]
    # features[POS_VBG] = [len(list(filter(lambda x: x[1][:3] == 'VBG', a))) for a in pos_tags]
    # features[POS_VBN] = [len(list(filter(lambda x: x[1][:3] == 'VBN', a))) for a in pos_tags]
    # features[POS_VBP] = [len(list(filter(lambda x: x[1][:3] == 'VBP', a))) for a in pos_tags]
    # features[POS_VBZ] = [len(list(filter(lambda x: x[1][:3] == 'VBZ', a))) for a in pos_tags]
    features[POS_UH] = [len(list(filter(lambda x: x[1][:2] == 'UH', a))) for a in pos_tags]
    features[POS_JJ] = [len(list(filter(lambda x: x[1][:2] == 'JJ', a))) for a in pos_tags]
    features[POS_RP] = [len(list(filter(lambda x: x[1][:2] == 'RP', a))) for a in pos_tags]
    features[POS_RB] = [len(list(filter(lambda x: x[1][:2] == 'RB', a))) for a in pos_tags]
    features[POS_WDT] = [len(list(filter(lambda x: x[1][:3] == 'WDT', a))) for a in pos_tags]
    features[POS_PR] = [len(list(filter(lambda x: x[1][:2] == 'PR', a))) for a in pos_tags]
    features[POS_DT] = [len(list(filter(lambda x: x[1][:2] == 'DT', a))) for a in pos_tags]
    features[POS_CC] = [len(list(filter(lambda x: x[1][:2] == 'CC', a))) for a in pos_tags]
    features[POS_CD] = [len(list(filter(lambda x: x[1][:2] == 'CD', a))) for a in pos_tags]

    # @ word
    features[AT_CAPS] = [len(list(filter(lambda x: x.startswith('CAPS'), a))) for a in words]
    features[AT_ORGANIZATION] = [len(list(filter(lambda x: x.startswith('ORGANIZATION'), a))) for a in words]
    features[AT_PERSON] = [len(list(filter(lambda x: x.startswith('PERSON'), a))) for a in words]
    features[AT_LOCATION] = [len(list(filter(lambda x: x.startswith('LOCATION'), a))) for a in words]
    features[AT_MONEY] = [len(list(filter(lambda x: x.startswith('MONEY'), a))) for a in words]
    features[AT_TIME] = [len(list(filter(lambda x: x.startswith('TIME'), a))) for a in words]
    features[AT_DATE] = [len(list(filter(lambda x: x.startswith('DATE'), a))) for a in words]
    features[AT_PERCENT] = [len(list(filter(lambda x: x.startswith('PERCENT'), a))) for a in words]
    features[AT_STATE] = [len(list(filter(lambda x: x.startswith('STATE'), a))) for a in words]
    features[AT_NUM] = [len(list(filter(lambda x: x.startswith('NUM'), a))) for a in words]
    features[AT_DR] = [len(list(filter(lambda x: x.startswith('DR'), a))) for a in words]
    features[AT_EMAIL] = [len(list(filter(lambda x: x.startswith('EMAIL'), a))) for a in words]
    features[AT_CITY] = [len(list(filter(lambda x: x.startswith('CITY'), a))) for a in words]
    features[AT_MONTH] = [len(list(filter(lambda x: x.startswith('MONTH'), a))) for a in words]

    return features


def concact_features(features_dict, keys=None):
    if not keys:
        keys = features_dict.keys()
    features = None
    for k in keys:
        if features is None:
            features = features_dict[k]
        else:
            features = np.c_[features, features_dict[k]]

    return features


def features_scalar(X):
    scaler = StandardScaler()

    scaler.fit(X)
    X = scaler.transform(X)

    LOGGER.info(f'X mean {scaler.mean_}')

    return scaler, X


def features_selector(X, y, k=10, need_features=None):
    selector = SelectKBest(f_regression, k=k)

    selector.fit(X, y)
    X = selector.transform(X)

    is_keep = selector.get_support(indices=tuple)
    scores = selector.scores_
    selected_features = []
    for i in is_keep:
        selected_features.append((need_features[i], scores[i]))
    selected_features.sort(key=lambda x: x[1], reverse=True)

    LOGGER.info(f'Select {k} features: {selected_features}')

    return selector, X


def get_tfidf_features(X, tfidf_max_features=100):
    stop_words = set(stopwords.words('english'))

    vectorizer = TfidfVectorizer(max_df=.2,
                                 min_df=3,
                                 max_features=100,
                                 stop_words=stop_words)
    X = vectorizer.fit_transform(X)

    return X.toarray(), vectorizer


def get_features(X, y, need_features=None, k=10, use_tfidf=False, use_fea_select=True):
    features_dict = get_all_features(X)
    if need_features is None:
        need_features = list(features_dict.keys())
    X_features = concact_features(features_dict, need_features)

    scalar_pure, X_features = features_scalar(X_features)

    selector = None
    if use_fea_select:
        selector, X_features = features_selector(X_features, y, k=k, need_features=need_features)

    vectorizer = None
    scalar_tfidf = None
    if use_tfidf:
        tfidf, vectorizer = get_tfidf_features(X)
        X_features = np.c_[X_features, tfidf]
        scalar_tfidf, X_features = features_scalar(X_features)

    return X_features, scalar_pure, scalar_tfidf, selector, vectorizer


def get_features_test(X, scalar_pure, scalar_tfidf, selector, vectorizer):
    features_dict = get_all_features(X)

    need_features = list(features_dict.keys())

    X_features = concact_features(features_dict, need_features)

    X_features = scalar_pure.transform(X_features)

    if selector is not None:
        X_features = selector.transform(X_features)

    if scalar_tfidf is not None:
        X_tfidf = vectorizer.transform(X)
        X_tfidf = X_tfidf.toarray()

        X_features = np.c_[X_features, X_tfidf]
        scalar, X_features = features_scalar(X_features)

    return X_features


if __name__ == '__main__':
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('averaged_perceptron_tagger')
    from integration.dataset import AESDataset
    # D = AESDataset(r'../essay_data/DEMO')
    D = AESDataset(ROOT_DIR + '/essay_data/DEMO')

    x_train, y_train = D.get_train()

    X, y = x_train[0][:100], y_train[0][:100]

    # features_dict = get_all_features(X)
    # X_features = concact_features(features_dict)
    # _, X_features = features_scalar(X_features)
    # _, selected_features = features_selector(X_features, y)
    # get_tfidf_features(X)
    X_features, scalar_pure, scalar_tfidf, selector, vectorizer = get_features(X, y, None, 15, True)

    print(X_features.shape)

    print(X_features[:10])