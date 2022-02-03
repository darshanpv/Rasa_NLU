# -*- coding: utf-8 -*-
import re
import json
from utils import nlp_config
from utils import log_util
from core.classifiers.tfidf import predict_tfidf
from core.classifiers.nlu import predict_nlu
from warnings import simplefilter

# ignore all warnings
simplefilter(action='ignore')


def predict(domain, locale, userUtterance):
    response = json.loads('{"response":"ERROR: error during predicting the user utterance"}')

    if not nlp_config.check_data_available(domain,locale):
        log_util.log_errormsg("[PREDICT_MODEL] no intent data found. Exiting...")
        return json.loads('{"response":"ERROR: no intent data found. Exiting..."}')

    if re.search(nlp_config.get_parameter('ENSEMBLE'), 'true', re.IGNORECASE):
        if nlp_config.get_parameter('ALGORITHM') == 'TFIDF':
            # make tfidf as base algorithm & normalise the entity score
            response_1 = predict_tfidf.predict(domain, locale, userUtterance)
            response_1 = nlp_config.normalise_entity_score(response_1)
            score_1 = nlp_config.get_scores(response_1)

            response_2 = predict_nlu.predict(domain, locale, userUtterance)
            response_2 = nlp_config.normalise_entity_score(response_2)
            score_2 = nlp_config.get_scores(response_2)
            # ensemble the scores
            response = nlp_config.ensemble_confidence_score(response_1, response_2)
            score = nlp_config.get_scores(response)
            log_util.log_infomsg(f'[PREDICT_MODEL] TFIDF scores: {score_1}')
            log_util.log_infomsg(f'[PREDICT_MODEL] NLU scores: {score_2}')
            log_util.log_infomsg(f'[PREDICT_MODEL] ENSEMBLE scores: {score}')

        elif nlp_config.get_parameter('ALGORITHM') == 'NLU':
            # make nlu as base algorithm & normalise the entity score
            response_1 = predict_nlu.predict(domain, locale, userUtterance)
            response_1 = nlp_config.normalise_entity_score(response_1)
            score_1 = nlp_config.get_scores(response_1)

            response_2 = predict_tfidf.predict(domain, locale, userUtterance)
            response_2 = nlp_config.normalise_entity_score(response_2)
            score_2 = nlp_config.get_scores(response_2)
            # ensemble the scores
            response = nlp_config.ensemble_confidence_score(response_1, response_2)
            score = nlp_config.get_scores(response)
            log_util.log_infomsg(f'[PREDICT_MODEL] NLU scores: {score_1}')
            log_util.log_infomsg(f'[PREDICT_MODEL] TFIDF scores: {score_2}')
            log_util.log_infomsg(f'[PREDICT_MODEL] ENSEMBLE scores: {score}')
    else:
        if nlp_config.get_parameter('ALGORITHM') == 'TFIDF':
            response = predict_tfidf.predict(domain, locale, userUtterance)
            # normalise the entity score
            response = nlp_config.normalise_entity_score(response)
            log_util.log_infomsg(f'[PREDICT_MODEL] TFIDF scores: {nlp_config.get_scores(response)}')
        elif nlp_config.get_parameter('ALGORITHM') == 'NLU':
            response = predict_nlu.predict(domain, locale, userUtterance)
            # normalise the entity score
            response = nlp_config.normalise_entity_score(response)
            log_util.log_infomsg(f'[PREDICT_MODEL] NLU scores: {nlp_config.get_scores(response)}')
        else:
            log_util.log_errormsg("[PREDICT_MODEL] configured algorithm is not supported. Exiting...")

    #result = json.dumps(json.load(response))
    return response


if __name__ == "__main__":
    predict("trip", "en", "I want to book a ticket")
