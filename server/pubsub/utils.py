# -*- coding: utf-8 -*-
import logging
import re
import sys


# this function will return partition number based on key value for proper distribution of topic records across
# partitions
def get_partition(key, partitionCount):
    return parse_key(key) % partitionCount


def parse_key(key):
    pNumber = -1
    p = re.compile("(d|s)(.*?)-")
    m = p.match(key)
    if m:
        pNumber = m.group(0)[1:-1]
    return int(pNumber)
