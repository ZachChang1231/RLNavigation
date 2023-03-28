# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : gym_envs.py
# Time       ：2023/3/16 下午5:48
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

import gym
import numpy as np

from envs.multiprocessing_env import SubprocVecEnv


class GymEnv(object):
    def __init__(self, cfg, logger, multi=False):
        self.cfg = cfg
        self.logger = logger
        self.rnd = np.random.RandomState(cfg.seed)

        if multi:
            _env = [self._make_env() for _ in range(cfg.num_processes)]
            _env = SubprocVecEnv(_env)
        else:
            _env = gym.make(self.cfg.env_name)
        self._env = _env
        self.observation_space, self.action_space = _env.observation_space, _env.action_space

        if multi:
            self.seed()

    def seed(self):
        self._env.seed([self.cfg.seed + i for i in range(self.cfg.num_processes)])

    def step(self, action):
        return self._env.step(action)

    def render(self):
        self._env.render()

    def reset(self):
        return self._env.reset()

    def close(self):
        self._env.close()

    def _make_env(self):
        def _thunk():
            env = gym.make(self.cfg.env_name)
            return env
        return _thunk
