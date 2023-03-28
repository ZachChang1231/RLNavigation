# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : a2c_agent.py
# Time       ：2023/3/12 上午3:36
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class A2CAgent(object):
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        # self.scheduler = StepLR(self.optimizer, step_size=200000, gamma=0.1)

    def update(self, rollout):
        log_probs, returns, values = rollout.feed_for_agent()
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + self.cfg.value_loss_coef * critic_loss - self.cfg.entropy_coef * rollout.entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

        return actor_loss.item(), critic_loss.item()

