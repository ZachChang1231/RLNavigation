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
from itertools import islice

from config import config as cfg

device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_cuda else 'cpu')


class RolloutStorage(object):
    def __init__(self, obs_shape, num_outputs):
        self.obs = torch.zeros(cfg.num_steps + 1, cfg.num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            cfg.num_steps + 1, cfg.num_processes, cfg.hidden_size)
        self.rewards = torch.zeros(cfg.num_steps, cfg.num_processes, 1)
        self.value_preds = torch.zeros(cfg.num_steps + 1, cfg.num_processes, 1)
        self.returns = torch.zeros(cfg.num_steps + 1, cfg.num_processes, 1)
        self.action_log_probs = torch.zeros(cfg.num_steps, cfg.num_processes, 1)
        self.actions = torch.zeros(cfg.num_steps, cfg.num_processes, 1)
        self.actions = self.actions.long()
        self.actions_onehot = torch.zeros(cfg.num_steps, cfg.num_processes, num_outputs)
        self.actions_onehot = self.actions_onehot.long()
        self.masks = torch.ones(cfg.num_steps + 1, cfg.num_processes, 1)
        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(cfg.num_steps + 1, cfg.num_processes, 1)
        self.done_masks = torch.ones(cfg.num_steps + 1, cfg.num_processes, 1)

        self.action_pretrained_oh = torch.zeros(cfg.num_steps, cfg.num_processes, num_outputs)

        self.register = ["obs", "recurrent_hidden_states", "rewards", "value_preds", "returns", "action_log_probs",
                         "actions", "actions_onehot", "masks", "bad_masks", "done_masks", "action_pretrained_oh"]
        self.num_steps = cfg.num_steps
        self.step = 0

    def to_device(self):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.actions_onehot = self.actions_onehot.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.done_masks = self.done_masks.to(device)

        self.action_pretrained_oh = self.action_pretrained_oh.to(device)

    def insert(self, dic):
        for key, value in dic.items():
            assert key in self.register, "Key not defined!"
            if key == "obs":
                value = torch.from_numpy(value).float()
            if key == "rewards":
                value = torch.from_numpy(value).unsqueeze(1)
            if key in ["obs", "recurrent_hidden_states", "masks", "bad_masks"]:
                step = "self.step + 1"
            else:
                step = "self.step"
            exec("self.{}[{}].copy_(value)".format(key, step))

    def step_(self):
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.done_masks[0].copy_(self.done_masks[-1])

    def compute_returns(self, next_value, gamma=0.99):
        if cfg.use_proper_time_limits:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]) * \
                                     self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]


class DataWriter(object):
    def __init__(self):
        self.max_len = int(cfg.num_steps * max(cfg.save_interval, cfg.eval_interval, cfg.log_interval) * 1.2)
        mapping_keys = ["total_reward", "extrinsic_reward", "intrinsic_reward", "eval_reward", "actor_loss",
                        "critic_loss", "dist_entropy", "curiosity_loss", "imitate_loss"]
        self.mapping = dict()
        for key in mapping_keys:
            self.mapping[key] = deque(maxlen=self.max_len)
        self.best_reward = 0

    def insert(self, data_dict):
        for key, value in data_dict.items():
            assert key in self.mapping.keys(), 'Key not defined!'
            self.mapping[key].append(value)

    def __call__(self):
        return self.mapping

    def _to_list(self, key, interval, step=1):
        assert key in self.mapping.keys(), 'Key not defined!'
        length = len(self.mapping[key])
        return list(islice(self.mapping[key], length - interval * step, length))

    def get_episode_loss(self, interval):
        return np.mean(self._to_list("actor_loss", interval)), np.mean(self._to_list("critic_loss", interval))

    def get_curiosity_info(self, interval):
        episode_curiosity_loss = self._to_list("curiosity_loss", interval)
        intrinsic_reward = self._to_list("intrinsic_reward", interval, cfg.num_steps)
        extrinsic_reward = self._to_list("extrinsic_reward", interval, cfg.num_steps)
        return np.mean(episode_curiosity_loss), np.mean(intrinsic_reward), np.mean(extrinsic_reward), np.max(intrinsic_reward)

    def get_episode_reward(self, interval):
        reward = self._to_list("total_reward", interval, cfg.num_steps)
        return np.mean(reward), np.median(reward), np.min(reward), np.max(reward)
