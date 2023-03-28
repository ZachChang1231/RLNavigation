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
import numpy as np
import torch
from itertools import count
from tqdm import tqdm

from agent import A2CAgent
from envs import get_env
from model.network import Policy
from model.storage import RolloutStorage, DataWriter
from model.utils import print_line


class A2CModel(object):
    def __init__(self, cfg, logger, device):
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.env = get_env(cfg, logger)
        obs_shape = self.env.observation_space.shape
        num_inputs = obs_shape[0]
        num_outputs = self.env.action_space.n

        self.datawriter = DataWriter()
        self.model = Policy(
            obs_shape,
            num_outputs,
            cfg
        ).to(device)

    def run(self):
        raise NotImplementedError

    def load_data(self, save_dir):
        self.model.load_state_dict(torch.load(save_dir))

    def save_data(self, path):
        save_dir = os.path.join(self.cfg.model_path, path)
        os.makedirs(save_dir) if not os.path.exists(save_dir) else None
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'policy.pt'))


class Trainer(A2CModel):
    def __init__(self, cfg, logger, device):
        super().__init__(cfg, logger, device)
        self.tester = Tester(cfg, logger, device)
        self.envs = get_env(cfg, logger, multi=True)
        self.agent = A2CAgent(cfg, self.model)
        self.rollout = RolloutStorage()

    def run(self):
        if self.cfg.pretrained_path:
            self.logger.info("Loading pretrained data...")
            self.load_data(self.cfg.pretrained_path)
        else:
            self.logger.info("Training from the very beginning...")

        frame_idx = 0
        done = np.array([True for _ in range(self.cfg.num_processes)])
        state = self.envs.reset(
            position=self.cfg.init_position,
            target_position=self.cfg.target_position,
            velocity=self.cfg.init_velocity
        )

        hns = [torch.zeros(1, self.cfg.hidden_size).to(self.device) for _ in range(self.cfg.num_processes)] if self.cfg.recurrent else None

        while frame_idx < self.cfg.max_frames:

            if self.cfg.recurrent:
                for i, v in enumerate(done):
                    if v:
                        hns[i] = torch.zeros(1, self.cfg.hidden_size).to(self.device)
                    else:
                        hns[i] = hns[i].detach()

            for _ in range(self.cfg.num_steps):
                state = torch.FloatTensor(state).to(self.device)
                dist, value, hns = self.model(state, hns)

                action = dist.sample()
                next_state, reward, terminated, truncated,  _ = self.envs.step(action.cpu().numpy())
                self.datawriter.insert({'total_reward': reward})

                self.rollout.insert(dist, action, value, reward, terminated, truncated)

                state = next_state

            frame_idx += 1

            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value, _ = self.model(next_state, hns)
            self.rollout.compute_returns(next_value, gamma=self.cfg.gamma)

            actor_loss, critic_loss = self.agent.update(self.rollout)

            if self.cfg.noise:
                self.model.reset_noise()

            self.datawriter.insert({'actor_loss': actor_loss, 'critic_loss': critic_loss})

            self.rollout.after_update()

            if frame_idx == self.cfg.max_frames - 1 and self.cfg.model_path != "":
                mean_r, _, _, _ = self.datawriter.get_episode_reward(self.cfg.save_interval)
                self.save_data("{}_avg_reward_{:.1f}".format(frame_idx, mean_r))

            if frame_idx % self.cfg.log_interval == 0:
                episode_actor_loss, episode_critic_loss = self.datawriter.get_episode_loss(self.cfg.log_interval)
                mean_r, median_r, min_r, max_r = self.datawriter.get_episode_reward(self.cfg.log_interval)
                self.logger.info("Num frame index: {}/{}, "
                                 "Last {} training episode: actor loss: {:.2f}, critic loss: {:.2f}, "
                                 "mean/median reward: {:.1f}/{:.1f}, "
                                 "min/max reward: {:.1f}/{:.1f}".format(frame_idx, int(self.cfg.max_frames),
                                                                        self.cfg.log_interval,
                                                                        episode_actor_loss, episode_critic_loss,
                                                                        mean_r, median_r, min_r, max_r))

            if self.cfg.eval_interval is not None and frame_idx % self.cfg.eval_interval == 0:

                print_line(self.logger, 'evaluate')
                self.tester.model.load_state_dict(self.model.state_dict())
                eval_reward = np.mean([self.tester.run() for _ in range(self.cfg.eval_num)])
                self.datawriter.insert({'eval_reward': eval_reward})
                self.logger.info("Num frame index: {}/{}, Eval reward: {:.2f}".format(frame_idx,
                                                                                      int(self.cfg.max_frames),
                                                                                      eval_reward))

                if eval_reward > self.datawriter.best_reward:
                    self.datawriter.best_reward = eval_reward
                    self.save_data("{}_avg_reward_{:.1f}".format(frame_idx, eval_reward))
                    self.logger.info("New best model saved!")

                if frame_idx < self.cfg.max_frames:
                    print_line(self.logger, 'train')

        return self.datawriter.to_dist()


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
        hn = [torch.zeros(1, self.cfg.hidden_size).to(self.device)] if self.cfg.recurrent else None
        for i in count():
            if render:
                self.env.render()
            if save_path:
                self.env.save_fig(i, save_path)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _, hn = self.model(state, hn)
            next_state, reward, terminated, truncated, _ = self.env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

    def seed(self, seed):
        self.env.seed(seed)
