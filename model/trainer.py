# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : trainer.py
# Time       ：2023/3/12 上午3:37
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
from tqdm import tqdm

from agent import A2CAgent
from envs import get_env
from model.network import Policy, IntrinsicCuriosityModule
from model.storage import RolloutStorage, DataWriter
from model.utils import print_line


class A2CModel(object):
    def __init__(self, cfg, logger, device):
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.env = get_env(cfg, logger)
        self.obs_shape = self.env.observation_space.shape
        self.num_inputs = self.obs_shape[0]
        self.num_outputs = self.env.action_space.n

        self.datawriter = DataWriter()
        self.model = Policy(
            self.obs_shape,
            self.num_outputs,
            cfg
        ).to(device)
        if cfg.icm:
            self.icm = IntrinsicCuriosityModule(
                self.obs_shape,
                self.num_outputs,
                cfg
            ).to(device)
        else:
            self.icm = None

    def run(self):
        raise NotImplementedError

    def load_data(self, save_dir):
        self.model.load_state_dict(torch.load(os.path.join(save_dir, "policy.pt")))
        if self.cfg.icm:
            self.icm.load_state_dict(torch.load(os.path.join(save_dir, "icm.pt")))

    def save_data(self, path):
        save_dir = os.path.join(self.cfg.model_path, path)
        os.makedirs(save_dir) if not os.path.exists(save_dir) else None
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'policy.pt'))
        if self.cfg.icm:
            torch.save(self.icm.state_dict(), os.path.join(save_dir, 'icm.pt'))


class Trainer(A2CModel):
    def __init__(self, cfg, logger, device):
        super().__init__(cfg, logger, device)
        self.model.train()
        self.tester = Tester(cfg, logger, device)
        self.envs = get_env(cfg, logger, multi=True)
        self.rollout = RolloutStorage(self.obs_shape, self.num_outputs)
        if cfg.icm:
            self.icm.train()
            self.fwd_criterion = nn.MSELoss(reduction="none")
            self.eta = self.cfg.eta
        self.agent = A2CAgent(cfg, self.model, self.icm)

    def run(self):
        if self.cfg.pretrained_path:
            self.logger.info("Loading pretrained data...")
            self.load_data(self.cfg.pretrained_path)
        else:
            self.logger.info("Training from the very beginning...")

        frame_idx = 0
        save_iter = 0

        state = self.envs.reset(
            position=self.cfg.init_position,
            target_position=self.cfg.target_position,
            velocity=self.cfg.init_velocity
        )
        state = torch.from_numpy(state).float().to(self.device)
        self.rollout.obs[0].copy_(state)
        self.rollout.to_device()
        start = time.time()

        while frame_idx < self.cfg.max_frames:

            for step in range(self.cfg.num_steps):
                with torch.no_grad():
                    dist, value, hns = self.model(self.rollout.obs[step], self.rollout.recurrent_hidden_states[step],
                                                  self.rollout.masks[step])
                action = dist.sample()
                action_log_probs = dist.log_probs(action)
                action_np = action.cpu().numpy().squeeze()
                next_state, reward, terminated, truncated, _ = self.envs.step(action_np)
                masks = torch.FloatTensor([[0.0] if terminated_ else [1.0] for terminated_ in terminated])
                bad_masks = torch.FloatTensor([[0.0] if truncated_ else [1.0] for truncated_ in truncated])
                action_oh = F.one_hot(torch.from_numpy(action_np), num_classes=self.num_outputs)
                self.rollout.insert(
                    {
                        "obs": next_state,
                        "recurrent_hidden_states": hns,
                        "actions": action,
                        "actions_onehot": action_oh,
                        "action_log_probs": action_log_probs,
                        "value_preds": value,
                        "masks": masks,
                        "bad_masks": bad_masks
                    }
                )
                if self.cfg.icm:
                    with torch.no_grad():
                        pred_logits, pred_phi, phi = self.icm(self.rollout.obs[step],
                                                              self.rollout.obs[step + 1],
                                                              self.rollout.actions_onehot[step])
                        fwd_loss = self.fwd_criterion(pred_phi, phi).mean(dim=1) / 2
                        intrinsic_reward = self.eta * fwd_loss.detach()
                        reward = reward + intrinsic_reward.cpu().numpy()
                self.rollout.insert({"rewards": reward})
                self.rollout.step_()
                self.datawriter.insert({'total_reward': reward})

            frame_idx += 1
            save_iter += 1
            with torch.no_grad():
                _, next_value, _ = self.model(self.rollout.obs[-1], self.rollout.recurrent_hidden_states[-1],
                                              self.rollout.masks[-1])
            self.rollout.compute_returns(next_value.detach(), self.cfg.gamma)
            value_loss, action_loss, dist_entropy, curiosity_loss = self.agent.update(self.rollout)
            self.rollout.after_update()

            if self.cfg.noise:
                self.model.reset_noise()

            if self.cfg.eta_decay:
                self.eta -= self.cfg.eta / self.cfg.max_frames

            self.datawriter.insert({"actor_loss": action_loss, "critic_loss": value_loss,
                                    "dist_entropy": dist_entropy, "curiosity_loss": curiosity_loss})

            if frame_idx == self.cfg.max_frames - 1 and self.cfg.model_path != "":
            # if save_iter >= self.cfg.save_interval and self.cfg.model_path != "":
                mean_r, _, _, _ = self.datawriter.get_episode_reward(self.cfg.save_interval)
                self.save_data("{}_avg_reward_{:.1f}".format(frame_idx, mean_r))
                save_iter = 0

            if frame_idx % self.cfg.log_interval == 0:
                end = time.time()
                time_cost = end - start
                start = end
                episode_actor_loss, episode_critic_loss, episode_curiosity_loss = \
                    self.datawriter.get_episode_loss(self.cfg.log_interval)
                mean_r, median_r, min_r, max_r = self.datawriter.get_episode_reward(self.cfg.log_interval)
                self.logger.info("Num frame index: {}/{}, "
                                 "Last {} training episode: "
                                 "actor loss: {:.2f}, critic loss: {:.2f}, curiosity loss: {}, "
                                 "mean/median reward: {:.1f}/{:.1f}, "
                                 "min/max reward: {:.1f}/{:.1f}, "
                                 "time cost: {:.1f}s".format(frame_idx, int(self.cfg.max_frames),
                                                             self.cfg.log_interval,
                                                             episode_actor_loss, episode_critic_loss,
                                                             episode_curiosity_loss,
                                                             mean_r, median_r, min_r, max_r, time_cost))

            if self.cfg.eval_interval is not None and frame_idx % self.cfg.eval_interval == 0:

                print_line(self.logger, 'evaluate')
                self.tester.model.load_state_dict(self.model.state_dict())
                eval_reward = np.mean([self.tester.run() for _ in range(self.cfg.eval_num)])
                self.datawriter.insert({'eval_reward': eval_reward})
                self.logger.info("Num frame index: {}/{}, Eval reward: {:.2f}".format(frame_idx,
                                                                                      int(self.cfg.max_frames),
                                                                                      eval_reward))

                if eval_reward > self.datawriter.best_reward:
                    save_iter = 0
                    self.datawriter.best_reward = eval_reward
                    self.save_data("{}_avg_reward_{:.1f}".format(frame_idx, eval_reward))
                    self.logger.info("New best model saved!")

                if frame_idx < self.cfg.max_frames:
                    print_line(self.logger, 'train')

        return self.datawriter.to_dict()


class Tester(A2CModel):
    def __init__(self, cfg, logger, device):
        super().__init__(cfg, logger, device)
        self.model.eval()

    def run(self, render=False, save_path=""):
        state = self.env.reset(
            position=self.cfg.init_position,
            target_position=self.cfg.target_position,
            velocity=self.cfg.init_velocity
        )
        total_reward = 0
        hn = torch.zeros(1, self.cfg.hidden_size).to(self.device)
        mask = torch.zeros(1, 1).to(self.device)
        for i in count():
            if render:
                self.env.render()
            if save_path:
                self.env.save_fig(i, save_path)
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                dist, _, _ = self.model(state, hn, mask)
            action = dist.mode()
            next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0, 0])
            mask = torch.tensor(
                [[0.0] if terminated else [1.0]],
                dtype=torch.float32,
                device=self.device)
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

    def seed(self, seed):
        self.env.seed(seed)
