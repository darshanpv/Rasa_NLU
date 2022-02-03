# -*- coding: utf-8 -*-
import json
from utils import nlp_config
from utils import log_util
from kafka import KafkaProducer
from pubsub import utils

producer = None


def initialise():
    global producer
    producer = KafkaProducer(bootstrap_servers=nlp_config.get_parameter('KAFKA_BROKERS').split(","),
                             client_id=nlp_config.get_parameter('CLIENT_ID'),
                             value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                             linger_ms=1000,
                             retries=1
                             )


def send_message(topicName, key, value):
    global producer
    pNum = utils.get_partition(key, int(nlp_config.get_parameter('PARTITIONS')))
    msg = json.loads(value)
    producer.send(topicName, value=msg, key=key.encode('utf-8'), partition=pNum)
    producer.flush()
    log_util.log_infomsg("[PRODUCER] sending message: \"{}\"".format(value))
    log_util.log_infomsg("[PRODUCER] message sent with key: \"{}\" to partition \"{}\"!".format(key, pNum))
