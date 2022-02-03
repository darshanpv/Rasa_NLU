# -*- coding: utf-8 -*-
import os
import json
import re
import codecs
import pickle
from config import nluConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from warnings import simplefilter

# ignore all warnings
from coreNLP.modules import stemmer
from pubsub import utils

simplefilter(action='ignore')


def train(domain, locale):
    vectorDimension = int(nluConfig.getParameter('VECTOR_DIMENSION'))
    iterationNumbers = int(nluConfig.getParameter('ITERATION_NUMBER'))

    scriptDir = os.path.dirname(__file__)
    fileData = os.path.join(scriptDir, 'data', domain + '_' + locale + '.json')

    utterance = []
    intent = []
    try:
        with codecs.open(fileData, 'r', 'utf-8')as dataFile:
            data = json.load(dataFile)
    except FileNotFoundError:
        utils.loginfomsg("Error: domain file does not exist for NLP training.")
        res = {"intents": "0", "utterances": "0"}
        response = str(res).replace("'", '"').strip()
        return response

    for nameUtterances in data['tasks']:
        for utt in nameUtterances['utterances']:
            utterance.append(utt)
            intent.append(nameUtterances['name'])

    mIntent = set(intent)

    stopListFile = os.path.join(scriptDir, 'dictionary', 'stopwords_' + locale + '.txt')

    arrayWords = []
    stopWords = []

    f = codecs.open(stopListFile, 'r', 'utf-8')
    lines = f.read().split("\n")
    for line in lines:
        if line != "":
            arrayWords.append(line.split(','))

    for a_word in arrayWords:
        for s_word in a_word:
            if (re.sub(' ', '', s_word)) != "":
                stopWords.append(s_word)

    extraStopWords = set(stopWords)
    if locale == 'ar':
        stops = set(stopwords.words('arabic')) | extraStopWords
    elif locale == 'da':
        stops = set(stopwords.words('danish')) | extraStopWords
    elif locale == 'en':
        stops = set(stopwords.words('english')) | extraStopWords
    elif locale == 'es':
        stops = set(stopwords.words('spanish')) | extraStopWords
    elif locale == 'hi':
        stops = extraStopWords
    elif locale == 'mr':
        stops = extraStopWords
    elif locale == 'nl':
        stops = set(stopwords.words('dutch')) | extraStopWords
    elif locale == 'sv':
        stops = set(stopwords.words('swedish')) | extraStopWords
    else:
        res = {"intents": "0", "utterances": "0"}
        response = str(res).replace("'", '"').strip()
        return response

    stemmer.setLocale(locale)

    tfidfVec = TfidfVectorizer(utterance, decode_error='ignore', stop_words=stops, ngram_range=(1, 5),
                               tokenizer=stemmer.stemTokenize)
    trainsetIdfVectorizer = tfidfVec.fit_transform(utterance).toarray()
    vLength = len(trainsetIdfVectorizer[1])
    nDimension = vectorDimension
    if vLength <= vectorDimension:
        nDimension = vLength - 1

    svd = TruncatedSVD(n_components=nDimension, algorithm='randomized', n_iter=iterationNumbers, random_state=42)
    trainLSA = svd.fit_transform(trainsetIdfVectorizer)

    pickle_path = os.path.join(scriptDir, 'model', domain + '_' + locale + '_')
    fileName = pickle_path + 'utterance.m'
    fileObject = open(fileName, 'wb')
    pickle.dump(utterance, fileObject)
    fileObject.close()
    fileName = pickle_path + 'intent.m'
    fileObject = open(fileName, 'wb')
    pickle.dump(intent, fileObject)
    fileObject.close()
    fileName = pickle_path + 'tfidfVec.m'
    fileObject = open(fileName, 'wb')
    pickle.dump(tfidfVec, fileObject)
    fileObject.close()
    fileName = pickle_path + 'svd.m'
    fileObject = open(fileName, 'wb')
    pickle.dump(svd, fileObject)
    fileObject.close()
    fileName = pickle_path + 'trainLSA.m'
    fileObject = open(fileName, 'wb')
    pickle.dump(trainLSA, fileObject)
    fileObject.close()

    utils.loginfomsg(f'Identified domain: {domain}')
    utils.loginfomsg(f'Identified locale: {locale}')
    utils.loginfomsg(f'Number of utterances for training: {len(intent)}')
    utils.loginfomsg(f'Number of intents for training: {len(mIntent)}')

    res = {"intents": str(len(intent)), "utterances": str(len(mIntent))}
    response = str(res).replace("'", '"').strip()  # make it a string
    return response
