# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import logging

myLogger = logging.getLogger('__name__')

def log_infomsg(msg):
    myLogger.info(msg)
def log_errormsg(msg):
    myLogger.error(msg)
