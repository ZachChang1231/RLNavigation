# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Time       ：2023/3/12 上午3:37
# Author     ：Zach
# version    ：python 3.8
# Description：
"""


def print_line(logger, message):
    if message == 'line':
        logger.info("----------------------------------------")
    elif message == 'evaluate':
        logger.info("----------------------------------------")
        logger.info("               Evaluating               ")
        logger.info("----------------------------------------")
    elif message == 'train':
        logger.info("----------------------------------------")
        logger.info("                Training                ")
        logger.info("----------------------------------------")
    elif message == 'test':
        logger.info("----------------------------------------")
        logger.info("                Testing                 ")
        logger.info("----------------------------------------")
    elif message == 'load':
        logger.info("----------------------------------------")
        logger.info("                Loading                 ")
        logger.info("----------------------------------------")
    else:
        raise ValueError('Message undefined!')


