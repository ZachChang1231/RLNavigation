# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : __init__.py
# Time       ：2023/3/16 下午5:46
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

from gym import envs

from envs.custom_envs import CusEnv, MultiCusEnv
from envs.gym_envs import GymEnv


def check_env(env_name):
    env_list = []
    for env in envs.registry.all():
        env_list.append(env.id)
    if env_name in env_list:
        return True
    else:
        return False


def get_env(cfg, logger, multi=False, **kwargs):
    if check_env(cfg.env_name):
        env = GymEnv(cfg, logger, multi)
    else:
        if multi:
            env = MultiCusEnv(cfg, logger)
        else:
            env = CusEnv(cfg, logger, **kwargs)
    return env



