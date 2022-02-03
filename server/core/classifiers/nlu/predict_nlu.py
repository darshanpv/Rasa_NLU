# -*- coding: utf-8 -*-
import os
import asyncio
from collections import OrderedDict
from utils import nlp_config
from warnings import simplefilter
from rasa.core.agent import Agent

from rasa.shared.nlu.training_data.loading import load_data
from rasa.shared.utils.io import json_to_string
from utils import log_util

scriptDir = os.path.dirname(__file__)
dataFile = ""

simplefilter(action='ignore')


def predict(domain, locale, userUtterance):
    modelPath = os.path.join(scriptDir, '..', '..', 'models', 'nlu')
    global dataFile
    dataFile = os.path.join(scriptDir, '..', '..', '..', 'training_data', 'intents', domain + '_' + locale + '.yml')
    MODEL_NAME = domain + '_' + locale

    agent = Agent.load(model_path=modelPath)
    #use of async loop for rasa
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    data = loop.run_until_complete(agent.parse_message(userUtterance))

    intent_, score_, utterance_ = [], [], []
    intent_.append(data['intent_ranking'][0]['name'])
    intent_.append(data['intent_ranking'][1]['name'])
    intent_.append(data['intent_ranking'][2]['name'])
    score_.append("{:.2f}".format(data['intent_ranking'][0]['confidence']))
    score_.append("{:.2f}".format(data['intent_ranking'][1]['confidence']))
    score_.append("{:.2f}".format(data['intent_ranking'][2]['confidence']))
    utterance_.append(getUtterance(intent_[0]))
    utterance_.append(getUtterance(intent_[1]))
    utterance_.append(getUtterance(intent_[2]))
    entities_ = data['entities']
    text_ = data['text']
    intent_ranking_ = [{"name": p, "confidence": q, "utterance": r} for p, q, r in zip(intent_, score_, utterance_)]
    intent_top_ = {"name": intent_[0], "confidence": score_[0]}
    # build JSON response
    response = {'intent': intent_top_, 'entities': entities_, 'intent_ranking': intent_ranking_, 'text': text_}
    log_util.log_infomsg(f"[PREDICT_NLU] prediction: {response}")
    return response


def getUtterance(intent_):
    train_data = load_data(dataFile)
    training_examples = OrderedDict()
    INTENT = 'intent'
    for example in [e.as_dict_nlu() for e in train_data.training_examples]:
        intent = example[INTENT]
        training_examples.setdefault(intent, [])
        training_examples[intent].append(example)
    return training_examples[intent_][0]['text']
