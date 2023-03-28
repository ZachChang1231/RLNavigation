# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : network.py
# Time       ：2023/3/12 上午3:37
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, obs_shape, num_outputs, cfg):
        super(Policy, self).__init__()
        self.cfg = cfg

        if len(obs_shape) == 3:
            base = CNNBase
        elif len(obs_shape) == 1:
            base = MLPBase
        else:
            raise NotImplementedError

        self.base = base(obs_shape[0], num_outputs, cfg)
        self.soft_max = nn.Softmax(dim=1)
        if not self.cfg.noise:
            self.actor_linear = nn.Linear(cfg.hidden_size, num_outputs)
            self.critic_linear = nn.Linear(cfg.hidden_size, 1)
        else:
            self.actor_linear = NoisyLinear(cfg.hidden_size, num_outputs, sigma_init=cfg.sigma_init)
            self.critic_linear = NoisyLinear(cfg.hidden_size, 1)

        self.init()
        self.train()

    def forward(self, inputs, hns):
        x, hns = self.base(inputs, hns)
        value = self.critic_linear(x)
        probs = self.soft_max(self.actor_linear(x))
        dist = Categorical(probs)

        return dist, value, hns

    def reset_noise(self):
        if self.cfg.noise:
            self.actor_linear.reset_noise()
            self.critic_linear.reset_noise()

    def remove_noise(self):
        if self.cfg.noise:
            self.actor_linear.remove_noise()
            self.critic_linear.remove_noise()

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def get_value(self, inputs, hn):
        value, _, _ = self.base(inputs, hn)
        return value

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _gru(self, x, hn):
        if x.size(0) == hn.size(0):
            x, hn = self.gru(x.unsqueeze(0), hn.unsqueeze(0))
            x = x.squeeze(0)
            hn = hn.squeeze(0)
        else:
            pass
            # TODO  Conv2d situation
        return x, hn


class CNNBase(NNBase):
    def __init__(self, num_inputs, num_outputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.fc = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, hidden_size),
            nn.ReLU()
        )

        self.actor_linear = nn.Sequential(
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1)
        )
        self.critic_linear = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        self.init()

        self.train()

    def forward(self, inputs, hn):
        # TODO
        x = self.fc(inputs)

        if self.is_recurrent:
            x, hn = self._gru(x, hn)

        value = self.critic_linear(x)
        probs = self.actor_linear(x)
        dist = Categorical(probs)

        return dist, value, hn


class MLPBase(NNBase):
    def __init__(self, num_inputs, num_outputs, cfg):
        super(MLPBase, self).__init__(cfg.recurrent, cfg.hidden_size, cfg.hidden_size)
        hidden_size = cfg.hidden_size

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, inputs, hns):
        x = self.fc(inputs)
        if self.is_recurrent:
            split_x = torch.split(x, 1, dim=0)
            x_list = []
            hns_list = []
            for x, hn in zip(split_x, hns):
                x, hn = self._gru(x, hn)
                x_list.append(x)
                hns_list.append(hn)
            x = torch.cat(x_list)
            hns = hns_list
        return x, hns


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def remove_noise(self):
        self.weight_epsilon.copy_(torch.zeros(self.out_features, self.in_features))
        self.bias_epsilon.copy_(torch.zeros(self.out_features))

