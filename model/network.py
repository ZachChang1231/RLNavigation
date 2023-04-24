# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : network.py
# Time       ：2023/3/12 上午3:37
# Author     ：Zach
# version    ：python 3.8
# Description：
"""
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
# from torch.distributions import Categorical

from model.distribution import FixedCategorical as Categorical
from model.utils import load_state_dict


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
        self.base = base(cfg, cfg.recurrent)
        hidden_size = cfg.hidden_size
        for module in reversed(list(self.base.fc_c.modules())):
            if "Linear" in str(module.__class__):
                hidden_size = module.out_features
                break
        if self.cfg.noise:
            self.actor_linear = nn.Sequential(
                # NoisyLinear(hidden_size, hidden_size, sigma_init=cfg.sigma_init),
                # nn.ReLU(),
                NoisyLinear(hidden_size, num_outputs, sigma_init=cfg.sigma_init),
                nn.Softmax(dim=1),
            )
        else:
            self.actor_linear = nn.Sequential(
                # nn.Linear(hidden_size, hidden_size),
                # nn.ReLU(),
                nn.Linear(hidden_size, num_outputs),
                nn.Softmax(dim=1),
            )
        self.critic_linear = nn.Sequential(
            # nn.Linear(cfg.hidden_size, cfg.hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        if cfg.last_n:
            hold_list = ["actor_module", "critic_module"]
            self._hold_parameter(hold_list)

        self.init()

    def forward(self, inputs, hns, masks):
        x, hns = self.base(inputs, hns, masks)
        value = self.critic_linear(x)
        probs = self.actor_linear(x)
        dist = Categorical(logits=probs)
        return dist, value, hns

    def evaluate_actions(self, inputs, hns, action, masks):
        x, hns = self.base(inputs, hns, masks)
        value = self.critic_linear(x)
        probs = self.actor_linear(x)
        dist = Categorical(logits=probs)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, hns

    def reset_noise(self):
        if self.cfg.noise:
            for module in self.modules():
                if "NoisyLinear" in str(module.__class__):
                    module.reset_noise()

    def remove_noise(self):
        if self.cfg.noise:
            for module in self.modules():
                if "NoisyLinear" in str(module.__class__):
                    module.remove_noise()

    def _hold_parameter(self, hold_list):
        for sequence in hold_list:
            module_list = []
            for name, para in self.named_parameters():
                if sequence in name:
                    module_list.append(name.split('.')[1])
            module_list = sorted([int(layer) for layer in list(set(module_list))])[::-1]
            for name, para in self.named_parameters():
                if not "{}.{}".format(sequence, module_list[self.cfg.last_n - 1]) in name:
                    para.requires_grad = False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

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
    def __init__(self, cfg, recurrent):
        super(NNBase, self).__init__()
        self.cfg = cfg
        self._hidden_size = cfg.hidden_size
        self._recurrent = recurrent

        if self._recurrent:
            self.gru = nn.GRU(cfg.hidden_size, cfg.hidden_size)
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

    def _gru(self, x, hn, masks):
        if x.size(0) == hn.size(0):
            x, hn = self.gru(x.unsqueeze(0), hn.unsqueeze(0))
            x = x.squeeze(0)
            hn = hn.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hn.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hn = hn.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hn * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hn = hn.squeeze(0)

        return x, hn


class CNNBase(NNBase):
    def __init__(self, cfg, recurrent):    # TODO
        super(CNNBase, self).__init__(cfg, recurrent)

        self.fc = nn.Sequential(
            nn.Conv2d(cfg.laser_num, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, cfg.hidden_size),
            nn.ReLU()
        )

    def forward(self, inputs, hn):
        x = self.fc(inputs)
        if self.is_recurrent:
            x, hn = self._gru(x, hn)

        return x, hn


class MLPBase(NNBase):
    def __init__(self, cfg, recurrent):
        super(MLPBase, self).__init__(cfg, recurrent)
        half_hidden_size = int(cfg.hidden_size / 2)
        quad_hidden_size = int(cfg.hidden_size / 4)
        if cfg.task == "coll_avoid":
            self.feature = None
            self.fc_o = nn.Sequential(
                nn.Linear(in_features=cfg.laser_num, out_features=half_hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=half_hidden_size, out_features=quad_hidden_size),
                nn.ReLU()
            )
            self.fc_d = nn.Sequential(
                nn.Linear(in_features=cfg.laser_num, out_features=half_hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=half_hidden_size, out_features=quad_hidden_size),
                nn.ReLU()
            )
            self.fc_c = nn.Sequential(
                nn.Linear(in_features=half_hidden_size, out_features=half_hidden_size),
                nn.ReLU()
            )
        elif cfg.task == "offline":
            self.feature = None
            self.fc_c = nn.Sequential(
                nn.Linear(in_features=2, out_features=quad_hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=quad_hidden_size, out_features=quad_hidden_size),
                nn.ReLU()
            )
        elif cfg.task == "online":
            assert cfg.coll_avoid_pretrained_path and cfg.offline_pretrained_path, "Pretrained module not found!"
            cfg_online = copy.deepcopy(cfg)
            cfg_online.task = "coll_avoid"
            self.coll_avoid_module = MLPBase(cfg, cfg.recurrent)
            load_state_dict(self.coll_avoid_module, cfg.coll_avoid_pretrained_path)
            cfg_online.task = "offline"
            self.offline_module = MLPBase(cfg, cfg.recurrent)
            load_state_dict(self.offline_module, cfg.offline_pretrained_path)
            self.fc_c = nn.Sequential(
                nn.Linear(in_features=quad_hidden_size + half_hidden_size, out_features=half_hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=half_hidden_size, out_features=half_hidden_size),
                nn.ReLU()
            )
        else:
            raise NotImplementedError

    def forward(self, inputs, hns=None, masks=None, return_hn=True):
        if self.cfg.task == "coll_avoid":
            obs_feature = self.fc_o(inputs[:, :self.cfg.laser_num])
            delta_feature = self.fc_d(inputs[:, :self.cfg.laser_num] - inputs[:, self.cfg.laser_num:self.cfg.laser_num * 2])
            x = self.fc_c(torch.cat((obs_feature, delta_feature), dim=1))
            self.feature = x.detach()
        elif self.cfg.task == "offline":
            x = self.fc_c(inputs[:, -2:])
            self.feature = x.detach()
        elif self.cfg.task == "online":
            self.coll_avoid_module(inputs)
            self.offline_module(inputs)
            x = self.fc_c(torch.cat((self.coll_avoid_module.feature, self.offline_module.feature), dim=1))
        else:
            raise NotImplementedError

        # target_feature = self.fc_t(inputs[:, -2:])
        if self.is_recurrent:  # TODO
            x, hns = self._gru(x, hns, masks)
        if return_hn:
            return x, hns
        else:
            return x


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

        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self):
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


class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, obs_shape, num_outputs, cfg):
        super(IntrinsicCuriosityModule, self).__init__()
        self.cfg = cfg

        half_hidden_size = int(cfg.hidden_size / 2)
        quad_hidden_size = int(cfg.hidden_size / 4)
        self.base_spatial = nn.Sequential(
            nn.Linear(in_features=cfg.laser_num, out_features=half_hidden_size),
            nn.ReLU(),
            # nn.Linear(in_features=half_hidden_size, out_features=half_hidden_size),
            # nn.ReLU()
        )
        self.base_temporal = nn.Sequential(
            nn.Linear(in_features=2, out_features=quad_hidden_size),
            nn.ReLU(),
            # nn.Linear(in_features=quad_hidden_size, out_features=quad_hidden_size),
            # nn.ReLU()
        )
        self.base_fc = nn.Sequential(
            nn.Linear(in_features=half_hidden_size + quad_hidden_size, out_features=cfg.hidden_size),
            nn.ReLU(),
            # nn.Linear(in_features=half_hidden_size + quad_hidden_size, out_features=cfg.hidden_size),
            # nn.ReLU()
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.icm_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cfg.icm_hidden_size, num_outputs)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(cfg.hidden_size + num_outputs, cfg.icm_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cfg.icm_hidden_size, cfg.hidden_size)
        )

        self.init()

    def base_forward(self, state):
        obs_feature = self.base_spatial(state[:, :self.cfg.laser_num])
        target_feature = self.base_temporal(state[:, -2:])
        return self.base_fc(torch.cat((obs_feature, target_feature), dim=1))

    def forward(self, state, next_state, action):
        state_ft = self.base_forward(state)
        next_state_ft = self.base_forward(next_state)
        state_ft = state_ft.view(-1, self.cfg.hidden_size)
        next_state_ft = next_state_ft.view(-1, self.cfg.hidden_size)
        return self.inverse_net(torch.cat((state_ft, next_state_ft), 1)), self.forward_net(
            torch.cat((state_ft.detach(), action), 1)), next_state_ft

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
