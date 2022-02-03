# -*- coding: utf-8 -*-
from utils import nlp_config
from utils import log_util
from kafka.admin import KafkaAdminClient, NewTopic


def create():
    admin_client = KafkaAdminClient(bootstrap_servers=nlp_config.get_parameter('KAFKA_BROKERS').split(","),
                                    client_id=nlp_config.get_parameter('CLIENT_ID'))
    topic_list = []
    log_util.log_infomsg("[CREATE_TOPIC] creating topic: \"{}\" with partition \"{}\" and replicas \"{}\"".format(
        nlp_config.get_parameter('TOPIC_BOT_TO_NLP'), nlp_config.get_parameter('PARTITIONS'),
        nlp_config.get_parameter('REPLICAS')))
    topic_list.append(NewTopic(name=nlp_config.get_parameter('TOPIC_BOT_TO_NLP'),
                               num_partitions=int(nlp_config.get_parameter('PARTITIONS')),
                               replication_factor=int(nlp_config.get_parameter('REPLICAS'))))

    log_util.log_infomsg("[CREATE_TOPIC] creating topic: \"{}\" with partition \"{}\" and replicas \"{}\"".format(
        nlp_config.get_parameter('TOPIC_NLP_TO_BOT'), nlp_config.get_parameter('PARTITIONS'),
        nlp_config.get_parameter('REPLICAS')))
    topic_list.append(NewTopic(name=nlp_config.get_parameter('TOPIC_NLP_TO_BOT'),
                               num_partitions=int(nlp_config.get_parameter('PARTITIONS')),
                               replication_factor=int(nlp_config.get_parameter('REPLICAS'))))
    admin_client.create_topics(new_topics=topic_list, validate_only=False)
