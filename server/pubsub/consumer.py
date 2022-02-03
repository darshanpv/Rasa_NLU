# -*- coding: utf-8 -*-
import json
from utils import nlp_config
from utils import log_util
from kafka import KafkaConsumer

consumer = None


def initialise(topic):
    global consumer
    consumer = KafkaConsumer(bootstrap_servers=nlp_config.get_parameter('KAFKA_BROKERS').split(","),
                             group_id=nlp_config.get_parameter('GROUP_ID'),
                             value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                             max_poll_records=int(nlp_config.get_parameter('MAX_POLL_RECORDS')),
                             auto_offset_reset=nlp_config.get_parameter('OFFSET_RESET'),
                             enable_auto_commit=True)
    consumer.subscribe(topic)
    # dummy poll to discard old messages
    consumer.poll()
    consumer.seek_to_end()
    return consumer


def read_messages(topic):
    global consumer
    consumer.subscribe(topic)
    for msg in consumer:
        log_util.log_infomsg(msg)
