# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : env_test.py
# Time       ：2023/3/22 下午9:45
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

import logging
import time
from itertools import count
import numpy as np

from config import config as cfg
from envs import get_env

if __name__ == "__main__":
    cfg.map_path = './envs/maps'
    cfg.env_name = 'map9'
    cfg.num_processes = 8

    cfg.shape_fixed = True
    cfg.render = True
    cfg.env_size = '500*500'

    logging.basicConfig(level=logging.INFO, filemode='w')
    logger = logging.getLogger(__name__)

    # envs = get_env(cfg, logger, multi=True)
    env = get_env(cfg, logger, multi=False, show=False)
    env.seed(np.random.randint(1e6))
    # env.add_moving_coll(20)
    # env.load_map(show=True)

    start = time.time()
    state = env.reset(position='10:20, 10:20', target_position='10:20, 10:20', velocity='1, 0')
    # state = env.reset()
    for t in count():
        if cfg.render:
            env.render()
        # action = [env.action_space.sample() for _ in range(cfg.num_processes)]
        action = env.action_space.sample()
        # state, reward, done, info = envs.step(action)
        state, reward, terminated, truncated, info = env.step(action)
        # if 1 in done or count > 1e3:
        if terminated or truncated:
            break
    env.close()
    logger.info("fps: {:.2f}".format(t / (time.time() - start)))
