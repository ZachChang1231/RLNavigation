# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Time       ：2023/3/12 上午3:37
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

import torch


class ActionScheduler(object):
    def __init__(self, cfg, action_num, test=False):
        self.cfg = cfg
        self.action_num = action_num
        self.k = 1
        self.lambda_ = 3.5
        self.phi_decay_steps = 1e6 * 2
        self.eps_decay_steps = 1e6
        # self.k = cfg.k
        # self.lambda_ = cfg.lambda_
        self.phi = 1
        self.eps = 0
        self.num = 1 if test else cfg.num_processes

    def __call__(self, action_online, action_coll, action_offline, obs, return_risk=False):
        mu = self._eval_risk(obs)
        action = torch.zeros(self.num, dtype=torch.long, device=action_online.device)
        for i in range(self.num):
            if torch.rand(1) < self.phi:
                if mu[i] < self.lambda_:
                    # offline
                    action[i].copy_(action_offline[i, 0])
                else:
                    # avoid
                    action[i].copy_(action_coll[i, 0])
            else:
                # online
                if torch.rand(1) < self.eps:
                    # random
                    action[i].copy_(torch.randint(self.action_num, (1,)).squeeze())
                else:
                    action[i].copy_(action_online[i, 0])
        self._step()
        if return_risk:
            return action.unsqueeze(1), mu.item()
        else:
            return action.unsqueeze(1)

    def _eval_risk(self, obs):
        ot = 1 - obs[:, :self.cfg.laser_num]
        dt = ot - (1 - obs[:, self.cfg.laser_num:self.cfg.laser_num * 2])
        mu = 1 / torch.min(ot - self.k * dt, dim=1)[0]
        return mu

    def _step(self):
        self.phi -= 1 / self.phi_decay_steps
        self.eps -= 1 / self.eps_decay_steps
        self.phi = max(0.0, self.phi)
        self.eps = max(0.0, self.eps)


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


def load_state_dict(model, path):
    pretrained_state_dict = torch.load(path)
    state_dict = dict()
    for key, value in pretrained_state_dict.items():
        if "base" in key:
            key_ = key.replace("base.", "")
            state_dict[key_] = value
    model.load_state_dict(state_dict)
