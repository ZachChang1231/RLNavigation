# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : trainer.py
# Time       ：2023/3/12 上午3:37
# Author     ：Zach
# version    ：python 3.8
# Description：
"""
import copy
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
from model.utils import print_line, ActionScheduler


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
        if cfg.task == "online":
            self.action_scheduler = ActionScheduler(cfg, action_num=self.num_outputs)
            cfg_coll, cfg_offline = copy.deepcopy(cfg), copy.deepcopy(cfg)
            cfg_coll.task = "coll_avoid"
            cfg_offline.task = "offline"
            self.coll_module = Policy(self.obs_shape, self.num_outputs, cfg_coll).to(device)
            self.offline_module = Policy(self.obs_shape, self.num_outputs, cfg_offline).to(device)
            self.coll_module.load_state_dict(torch.load(cfg.coll_avoid_pretrained_path))
            self.offline_module.load_state_dict(torch.load(cfg.offline_pretrained_path))

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
        start = time.time()
        try:
            result = self._run()
            self.logger.info("----------------------------------------")
            self.logger.info("        Done. Time cost: {:.2f}h        ".format((time.time() - start) / 3600))
            self.logger.info("----------------------------------------")
            return result
        except KeyboardInterrupt as e:
            self.logger.info("----------------------------------------")
            self.logger.info("      Interrupt. Time cost: {:.2f}h     ".format((time.time() - start) / 3600))
            self.logger.info("----------------------------------------")

    def _run(self):
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
                    dist, value, hns = self.model(self.rollout.obs[step],
                                                  self.rollout.recurrent_hidden_states[step],
                                                  self.rollout.masks[step])
                    if self.cfg.task == "online":
                        dist_coll, _, _ = self.coll_module(self.rollout.obs[step],
                                                           self.rollout.recurrent_hidden_states[step],
                                                           self.rollout.masks[step])
                        action_coll = dist_coll.mode()
                        dist_offline, _, _ = self.offline_module(self.rollout.obs[step],
                                                                 self.rollout.recurrent_hidden_states[step],
                                                                 self.rollout.masks[step])
                        action_offline = dist_offline.mode()
                action = dist.sample()
                if self.cfg.task == "online":
                    action = self.action_scheduler(action, action_coll, action_offline, self.rollout.obs[step])
                action_log_probs = dist.log_probs(action)
                action_np = action.cpu().numpy().squeeze()
                next_state, reward, terminated, truncated, _ = self.envs.step(action_np)
                self.datawriter.insert({"extrinsic_reward": reward})
                masks = torch.FloatTensor([[0.0] if terminated_ else [1.0] for terminated_ in terminated])
                bad_masks = torch.FloatTensor([[0.0] if truncated_ else [1.0] for truncated_ in truncated])
                done_masks = (masks.bool() & bad_masks.bool()).float()
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
                        "bad_masks": bad_masks,
                        "done_masks": done_masks,
                    }
                )
                if self.cfg.icm:
                    with torch.no_grad():
                        pred_logits, pred_phi, phi = self.icm(self.rollout.obs[step],
                                                              self.rollout.obs[step + 1],
                                                              self.rollout.actions_onehot[step])
                        fwd_loss = self.fwd_criterion(pred_phi, phi).mean(dim=1) * self.rollout.done_masks[
                            step].squeeze() / 2
                        intrinsic_reward = self.eta * fwd_loss.detach().cpu().numpy()
                        self.datawriter.insert({"intrinsic_reward": intrinsic_reward})
                        reward = reward + intrinsic_reward
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

            if self.cfg.eta_decay_steps:
                self.eta -= self.cfg.eta / self.cfg.eta_decay_steps
                self.eta = max(self.cfg.min_eta, self.eta)

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
                episode_actor_loss, episode_critic_loss = self.datawriter.get_episode_loss(self.cfg.log_interval)
                mean_r, median_r, min_r, max_r = self.datawriter.get_episode_reward(self.cfg.log_interval)
                self.logger.info("Num frame index: {}/{}, "
                                 "Last {} training episode: "
                                 "actor loss: {:.2f}, critic loss: {:.2f}, "
                                 "mean/median reward: {:.1f}/{:.1f}, "
                                 "min/max reward: {:.1f}/{:.1f}, "
                                 "time cost: {:.1f}s".format(frame_idx, int(self.cfg.max_frames),
                                                             self.cfg.log_interval,
                                                             episode_actor_loss, episode_critic_loss,
                                                             mean_r, median_r, min_r, max_r, time_cost))
                if self.cfg.icm:
                    curiosity_loss, intrinsic_reward, extrinsic_reward, max_in_r = \
                        self.datawriter.get_curiosity_info(self.cfg.log_interval)
                    self.logger.info("curiosity loss: {:.2f}, mean intrinsic/extrinsic reward: {:.4f}/{:.4f}, "
                                     "max intrinsic reward: {:.4f}, eta:{:.2f}".format(curiosity_loss,
                                                                                       intrinsic_reward,
                                                                                       extrinsic_reward,
                                                                                       max_in_r,
                                                                                       self.eta))
                if self.cfg.task == "online":
                    self.logger.info("phi: {:.2f}, eps: {:.2f}".format(
                        self.action_scheduler.phi,
                        self.action_scheduler.eps)
                    )

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

        return self.datawriter()


class Tester(A2CModel):
    def __init__(self, cfg, logger, device):
        super().__init__(cfg, logger, device)
        self.model.eval()
        if cfg.icm:
            self.icm.eval()
            self.fwd_criterion = nn.MSELoss()

    def run(self, render=False, save_path=""):
        self.seed(np.random.randint(1e6))
        state = self.env.reset(
            position=self.cfg.init_position,
            target_position=self.cfg.target_position,
            velocity=self.cfg.init_velocity
        )
        # state = self.env.reset(
        #     position="50, 200",
        #     target_position="750, 250",
        #     velocity=self.cfg.init_velocity,
        #     test=True
        # )
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        total_reward = 0
        hn = torch.zeros(1, self.cfg.hidden_size).to(self.device)
        mask = torch.zeros(1, 1).to(self.device)
        message = dict()
        for i in count():
            if render:
                self.env.render(message)
            if save_path:
                self.env.save_fig(i, save_path)
            with torch.no_grad():
                dist, _, _ = self.model(state, hn, mask)
            action = dist.mode()

            # action_scheduler = ActionScheduler(self.cfg, self.num_outputs, True)
            # with torch.no_grad():
            #     dist, value, hns = self.model(state, hn, mask)
            #     if self.cfg.task == "online":
            #         dist_coll, _, _ = self.coll_module(state, hn, mask)
            #         action_coll = dist_coll.mode()
            #         dist_offline, _, _ = self.offline_module(state, hn, mask)
            #         action_offline = dist_offline.mode()
            # action = dist.sample()
            # if self.cfg.task == "online":
            #     action, risk = action_scheduler(action, action_coll, action_offline, state, True)
            #     message["risk"] = risk

            action_oh = F.one_hot(action[0], num_classes=self.num_outputs)
            next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0, 0])
            message["extrinsic reward"] = reward
            mask = torch.tensor(
                [[0.0] if terminated else [1.0]],
                dtype=torch.float32,
                device=self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device).unsqueeze(0)
            if self.cfg.icm:
                with torch.no_grad():
                    pred_logits, pred_phi, phi = self.icm(state, next_state, action_oh)
                    fwd_loss = self.fwd_criterion(pred_phi, phi) / 2
                    intrinsic_reward = self.cfg.eta * fwd_loss.item()
                    message["intrinsic reward"] = intrinsic_reward
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

    def seed(self, seed):
        self.env.seed(seed)
