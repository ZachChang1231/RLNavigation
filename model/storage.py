# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : storage.py
# Time       ：2023/3/14 下午6:40
# Author     ：Zach
# version    ：python 3.8
# Description：
"""
import numpy as np
import torch
from collections import deque

from config import config as cfg

device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_cuda else 'cpu')


class RolloutStorage(object):
    def __init__(self):
        self.entropy = 0
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.bad_masks = []
        self.returns = []

    def insert(self, dist, action, value, reward, terminated, truncated):
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        self.entropy += entropy
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        self.masks.append(torch.FloatTensor(1 - terminated).unsqueeze(1).to(device))
        self.bad_masks.append(torch.FloatTensor(1 - truncated).unsqueeze(1).to(device))

    def after_update(self):
        self.__init__()

    def compute_returns(self, next_value, gamma=0.99):
        R = next_value
        for step in reversed(range(len(self.rewards))):
            if cfg.use_proper_time_limits:
                R = (self.rewards[step] + gamma * R * self.masks[step]) * self.bad_masks[step] + \
                    (1 - self.bad_masks[step]) * self.values[step]
            else:
                R = self.rewards[step] + gamma * R * self.masks[step]
            self.returns.insert(0, R)

    def feed_for_agent(self):
        log_probs = torch.cat(self.log_probs)
        returns = torch.cat(self.returns).detach()
        values = torch.cat(self.values)
        return log_probs, returns, values


class DataWriter(object):
    def __init__(self):
        self.mapping = {
            "total_reward": deque(maxlen=20000),
            "eval_reward": deque(maxlen=20000),
            "actor_loss": deque(maxlen=20000),
            "critic_loss": deque(maxlen=20000)
        }
        self.best_reward = 0

    def insert(self, data_dict):
        for key, value in data_dict.items():
            assert key in self.mapping.keys(), 'Key not defined!'
            self.mapping[key].append(value)

    def to_dist(self):
        return self.mapping

    def get_episode_loss(self, interval):
        return np.mean(self.mapping["actor_loss"][-interval:]), np.mean(self.mapping["critic_loss"][-interval:])

    def get_episode_reward(self, interval):
        reward = self.mapping["total_reward"][-interval*cfg.num_steps:]
        return np.mean(reward), np.median(reward), np.min(reward), np.max(reward)




