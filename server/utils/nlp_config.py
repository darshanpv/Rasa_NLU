# -*- coding: utf-8 -*-
import os
import json
import hashlib
from warnings import simplefilter
from utils import log_util

# ignore all warnings


simplefilter(action='ignore')

##Global parameters
scriptDir = os.path.dirname(__file__)

dataPath = os.path.join(scriptDir, '..', 'training_data', 'intents')
propertyFile = os.path.join(scriptDir, '..', 'config', 'nlp.properties')

separator = "="
properties = {}


def load_parameters() -> None:
    global properties
    with open(propertyFile) as f:
        for line in f:
            if separator in line:
                name, value = line.split(separator, 1)
                properties[name.strip()] = value.strip()


def getProperties():
    global properties
    return properties


def get_parameter(param):
    global properties
    res = ""
    if param in properties:
        res = properties[param]
        return res
    else:
        log_util.log_infomsg('[NLP_CONFIG] the required parameter could not be located'.format(param))
        return res


def check_data_available(domain, locale) -> bool:
    files = os.listdir(dataPath)
    for file in files:
        if file == (domain+'_'+ locale + '.yml'):
            return True
        else:
            pass
    return False


def is_config_stale(domain, locale):
    global properties
    tmpFile = os.path.join(scriptDir, '..', 'training_data', 'tmp', domain + '_hashdump')
    try:
        tmp = open(tmpFile, 'r')
    except IOError:
        tmp = open(tmpFile, 'a+')

    hash_original = tmp.read()
    # need to check if any changes to data, property file or rasa config file
    dataFile = os.path.join(dataPath, domain + '_' + locale + '.yml')
    data_1 = open(dataFile, 'rb').read()
    # check if any changs in properties
    load_parameters()
    data_2 = json.dumps(getProperties())
    if (get_parameter('ALGORITHM') == 'NLU'):
        rasaConfigFile = os.path.join(scriptDir, '..', 'core', 'config', get_parameter('CONFIG_FILE'))
        data_3 = open(rasaConfigFile, 'rb').read()
    else:
        data_3 = None
    totalData = str(data_1) + str(data_2) + str(data_3)
    hash_current = hashlib.md5(totalData.encode('utf-8')).hexdigest()
    if (hash_original == hash_current):
        return True
    else:
        tmp.close()
        tmp = open(tmpFile, 'w')
        tmp.write(hash_current)
        tmp.close()
        return False


def ensemble_confidence_score(response_1, response_2):
    scores_1 = get_scores(response_1)
    scores_2 = get_scores(response_2)
    # replace scores_1 with weighted average
    for item in scores_1:
        if item in scores_2:
            scores_1[item] = "{:.2f}".format((float(scores_1[item]) + float(scores_2[item])) / 2)
        else:
            scores_1[item] = "{:.2f}".format(float(scores_1[item]) / 2)

    # update the confidence score with new one
    for items in response_1["intent_ranking"]:
        if items["name"] in scores_1:
            items['confidence'] = scores_1[items["name"]]

    # update the intent JSONObject
    response_1["intent"]["confidence"] = scores_1[response_1["intent"]["name"]]
    return response_1


def get_scores(response):
    scores = {}
    for items in response["intent_ranking"]:
        scores[items["name"]] = items["confidence"]
    return scores


def normalise_entity_score(response):
    if len(response["entities"]) != 0:
        for items in response["entities"]:
            items["confidence_entity"] = "{:.3f}".format(items["confidence_entity"])
    return response
