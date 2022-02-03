# -*- coding: utf-8 -*-
import json
from utils import nlp_config
from utils import log_util
from core import train_model
from core import predict_model
from pubsub import utils
from pubsub import producer


def process(message):
    log_util.log_infomsg(
        '[PROCESS_MESSAGE]: message received with key: ' + message.key.decode('utf-8') + ' message: ' + str(
            message.value))
    key = message.key.decode('utf-8')
    # check if message has a correct key
    if utils.parse_key(key) == -1:
        log_util.log_infomsg('[PROCESS_MESSAGE] message contains wrong sessionID {}. Cannot process it'.format(key))
    # check if the message is for training the NLP
    elif utils.parse_key(key) == 0 and key.find('DUMMY') != -1:
        if 'messageId' in message.value and message.value['messageId'] == 'TRAIN':
            domain = message.value['domain']
            locale = message.value['locale']
            log_util.log_infomsg('[INTENT_ENGINE] training the NLP for domain:{} and locale:{}'.format(domain, locale))
            res = train_model.train(domain, locale)
            # convert res -> dictionary to String
            res_string = json.dumps(res)
            # convert res string to JSON
            res_json = json.loads(res_string)
            n = int(res_json["utterances"])
            if n > 0:
                response = {"messageId": "TRAIN_SUCCESS", "domain": domain, "locale": locale, "message": res_string}
            else:
                response = {"messageId": "TRAIN_FAIL", "domain": domain, "locale": locale, "message": res_string}
            producer.send_message(nlp_config.get_parameter('TOPIC_NLP_TO_BOT'), key, json.dumps(response))
        elif 'messageId' in message.value and message.value['messageId'] == 'PREDICT':
            domain = message.value['domain']
            locale = message.value['locale']
            utterance = message.value['userUtterance']
            log_util.log_infomsg(
                '[PROCESS_MESSAGE] predicting the utterance:{} for domain:{} and locale:{}'.format(utterance, domain,
                                                                                                   locale))
            result = predict_model.predict(domain, locale, utterance)
            response = {"messageId": "PREDICT", "domain": domain, "locale": locale, "userUtterance": utterance,
                        "message": result}
            producer.send_message(nlp_config.get_parameter('TOPIC_NLP_TO_BOT'), key, json.dumps(response))
    else:
        domain = message.value['domain']
        locale = message.value['locale']
        utterance = message.value['userUtterance']
        log_util.log_infomsg(
            '[PROCESS_MESSAGE] processing the utterance:{} for domain:{} and locale:{}'.format(utterance, domain,
                                                                                               locale))
        result = predict_model.predict(domain, locale, utterance)
        response = {"messageId": "PREDICT", "domain": domain, "locale": locale, "userUtterance": utterance,
                    "message": result}
        producer.send_message(nlp_config.get_parameter('TOPIC_NLP_TO_BOT'), key, json.dumps(response))
