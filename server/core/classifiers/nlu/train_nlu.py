# -*- coding: utf-8 -*-
import os
import asyncio
import shutil
from collections import OrderedDict
from utils import nlp_config
from rasa.model_training import train_nlu
from rasa.shared.nlu.training_data.loading import load_data
from utils import log_util
from warnings import simplefilter

simplefilter(action='ignore')

TRAINING_DATA = ''
CONFIG_DATA = ''
MODEL_NAME = ''

scriptDir = os.path.dirname(__file__)

def train(domain, locale):
    dataFile = os.path.join(scriptDir, '..', '..', '..', 'training_data', 'intents',
                            domain + '_' + locale + '.yml')
    configFile = os.path.join(scriptDir, '..', '..', 'config', nlp_config.get_parameter('CONFIG_FILE'))
    modelPath = os.path.join(scriptDir, '..', '..', 'models', 'nlu')
    model_name = domain + '_' + locale

    try:
        training_data = load_data(dataFile)
        # check if model file exists
        if os.path.exists(os.path.join(modelPath, model_name)):
            if not nlp_config.is_config_stale(domain, locale):
                # delete the folder if it exist
                if os.path.exists(os.path.join(modelPath, model_name)):
                    shutil.rmtree(os.path.join(modelPath, model_name))
                train_nlu(config=configFile, nlu_data=dataFile, output=modelPath, fixed_model_name=model_name, persist_nlu_training_data= True)
            else:
                log_util.log_infomsg("[TRAIN_NLU] no changes found to training data, using pre-trained model")
        else:  # train the model
             train_nlu(config=configFile, nlu_data=dataFile, output=modelPath, fixed_model_name=model_name, persist_nlu_training_data= True)
    except FileNotFoundError:
        log_util.log_errormsg("[TRAIN_NLU] could not locate the NLU config file")
        res = {"intents": "-1", "utterances": "-1"}
        return res

    training_examples = OrderedDict()
    INTENT = 'intent'
    for example in [e.as_dict_nlu() for e in training_data.training_examples]:
        intent = example[INTENT]
        training_examples.setdefault(intent, [])
        training_examples[intent].append(example)
    count = 0
    for x in training_examples:
        if isinstance(training_examples[x], list):
            count += len(training_examples[x])

    log_util.log_infomsg(f'[TRAIN_NLU] identified domain: {domain}')
    log_util.log_infomsg(f'[TRAIN_NLU] identified locale: {locale}')
    log_util.log_infomsg(f'[TRAIN_NLU] number of utterances for training: {count}')
    log_util.log_infomsg(f'[TRAIN_NLU] number of intents for training: {len(training_examples)}')

    algo = os.path.splitext(nlp_config.get_parameter('CONFIG_FILE'))[0]
    algo = algo.split("_")[1].upper()
    model = 'NLU:' + algo

    res = {"intents": str(len(training_examples)), "utterances": str(count), "model": model}
    return res
