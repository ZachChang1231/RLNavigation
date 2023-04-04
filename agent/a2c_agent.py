# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : a2c_agent.py
# Time       ：2023/3/12 上午3:36
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class A2CAgent(object):
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        # self.optimizer = optim.RMSprop(
        #     self.model.parameters(), lr=cfg.lr, eps=1e-5, alpha=0.99)
        # self.scheduler = StepLR(self.optimizer, step_size=200000, gamma=0.1)

    def update(self, rollout):
        obs_shape = rollout.obs.size()[2:]
        action_shape = rollout.actions.size()[-1]
        num_steps, num_processes, _ = rollout.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.model.evaluate_actions(
            rollout.obs[:-1].view(-1, *obs_shape),
            rollout.recurrent_hidden_states[0].view(
                -1, self.model.recurrent_hidden_state_size),
            rollout.actions.view(-1, action_shape),
            rollout.masks[:-1].view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollout.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.cfg.value_loss_coef + action_loss - dist_entropy * self.cfg.entropy_coef).backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

        self.optimizer.step()
        # self.scheduler.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

