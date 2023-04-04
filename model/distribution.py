# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : distribution.py
# Time       ：2023/3/31 下午6:00
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

from torch.distributions import Categorical


class FixedCategorical(Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)
