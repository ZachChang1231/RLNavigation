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
    def __init__(self, cfg, model, icm):
        self.cfg = cfg
        self.model = model
        critic = list(map(id, model.critic_linear.parameters()))
        base_params = filter(lambda p: id(p) not in critic, model.parameters())
        self.lr = cfg.lr * 10 if cfg.last_n else cfg.lr
        self.icm = icm
        self.fwd_criterion = nn.MSELoss(reduction="none")
        self.inv_criterion = nn.CrossEntropyLoss(reduction="none")
        self.cross_entropy = nn.CrossEntropyLoss()
        para = [{'params': base_params}, {'params': model.critic_linear.parameters(), 'lr': self.lr * 5}]
        if icm:
            para += [{'params': self.icm.parameters(), 'lr': self.lr * 1}]
        self.optimizer = optim.Adam(para, lr=self.lr)
        self.w = 1
        # self.optimizer = optim.RMSprop(
        #     self.model.parameters(), lr=cfg.lr, eps=1e-5, alpha=0.99)
        # self.scheduler = StepLR(self.optimizer, step_size=200000, gamma=0.1)

    def update(self, rollout):
        obs_shape = rollout.obs.size()[2:]
        action_shape = rollout.actions.size()[-1]
        num_steps, num_processes, _ = rollout.rewards.size()

        values, probs, action_log_probs, dist_entropy, _ = self.model.evaluate_actions(
            rollout.obs[:-1].view(-1, *obs_shape),
            rollout.recurrent_hidden_states[0].view(
                -1, self.model.recurrent_hidden_state_size),
            rollout.actions.view(-1, action_shape),
            rollout.masks[:-1].view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        probs = probs.view(num_steps, num_processes, -1)

        advantages = rollout.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        loss = value_loss * self.cfg.value_loss_coef + action_loss - dist_entropy * self.cfg.entropy_coef

        if self.cfg.imitate:
            imitate_loss = (self.cross_entropy(probs.transpose(1, 2),
                                               rollout.action_pretrained_oh.transpose(1, 2).float())).mean()
            loss = loss * (1 - self.w) + imitate_loss * self.w
            self.w = max(0.0, self.w - 1/1e5)
        else:
            self.w = 0

        curiosity_loss = None
        if self.cfg.icm:
            action_onehot_shape = rollout.actions_onehot.size()[-1]
            pred_logits, pred_phi, phi = self.icm(
                rollout.obs[:-1].view(-1, *obs_shape),
                rollout.obs[1:].view(-1, *obs_shape),
                rollout.actions_onehot.view(-1, action_onehot_shape))
            pred_logits = pred_logits.view(num_steps, num_processes, -1)
            pred_phi = pred_phi.view(num_steps, num_processes, -1)
            phi = phi.view(num_steps, num_processes, -1)
            done_masks = rollout.done_masks[:-1].squeeze()

            inv_loss = (self.inv_criterion(pred_logits.transpose(1, 2), rollout.actions_onehot.transpose(1, 2).float()) * done_masks).mean()
            fwd_loss = (self.fwd_criterion(pred_phi, phi).mean(dim=2) * done_masks).mean() / 2

            curiosity_loss = (inv_loss * (1 - self.cfg.beta) + self.cfg.beta * fwd_loss) * 10
            loss += curiosity_loss

        self.optimizer.zero_grad()
        loss.backward()
        para = list(self.model.parameters()) + list(self.icm.parameters()) if self.cfg.icm else list(self.model.parameters())
        nn.utils.clip_grad_norm_(para, self.cfg.max_grad_norm)
        self.optimizer.step()

        if self.cfg.icm:
            return value_loss.item(), action_loss.item(), dist_entropy.item(), curiosity_loss.item()
        # self.scheduler.step()
        else:
            return value_loss.item(), action_loss.item(), dist_entropy.item(), None

