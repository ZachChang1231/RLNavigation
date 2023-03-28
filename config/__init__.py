# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : __init__.py
# Time       ：2023/3/12 上午3:37
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

import os

from config.config import Config

config = Config()


def dump_config(path, _config):
    _config = config.to_dict()
    with open(os.path.join(path, 'config.txt'), 'w') as f:
        for key, value in _config.items():
            f.write("{}: {}\n".format(key, value))

