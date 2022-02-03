# -*- coding: utf-8 -*-
import re
from coreNLP.modules import processQuery as pq
from pubsub import utils


def predict(domain, locale, userUtterance):

    if locale == 'en':
        utter = re.sub(r'[^a-zA-Z ]', '', userUtterance)
    else:
        utter = userUtterance

    combinations = pq.genUtterances(utter,locale)
    jResult = pq.processUtterance(combinations,domain, locale)
    utils.loginfomsg(jResult)
    njResult = str(jResult).replace("'", '"').strip()
    #sys.stdout.buffer.write(newjResult.encode('utf8'))
    #print("\n")
    return njResult
    #return jResult