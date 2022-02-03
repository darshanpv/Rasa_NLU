# -*- coding: utf-8 -*-
import re
from core.classifiers.tfidf.utils import classifier
from utils import log_util


def predict(domain, locale, userUtterance):
    if locale == 'en':
        utter = re.sub(r'[^a-zA-Z ]', '', userUtterance)
    else:
        utter = userUtterance

    combinations = classifier.genUtterances(utter, locale)
    response = classifier.processUtterance(combinations, domain, locale)
    log_util.log_infomsg(f'[PREDICT_TFIDF]: {response}')
    return response
