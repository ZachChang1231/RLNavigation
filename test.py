# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : test.py
# Time       ：2023/3/12 上午3:36
# Author     ：Zach
# version    ：python 3.8
# Description：
"""


import logging
import os
import numpy as np
import torch
import imageio
import shutil

from config import config as cfg
from model.trainer import Tester
from model.utils import print_line


def create_gif(image_list, gif_name, delete_dir):
    frames = []
    for image_name in image_list:
        frames.append(imageio.v3.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.05)
    shutil.rmtree(delete_dir)


def main():
    # Set path
    save_path = os.getcwd()
    save_path = "{}/result/{}".format(save_path, time_stamp)
    if mode == "fig_saving":
        image_save_path = os.path.join(save_path, "temp")
        os.makedirs(image_save_path) if not os.path.exists(image_save_path) else None
    else:
        image_save_path = ""
    model_path = os.path.join(save_path, "checkpoint/{}".format(episode))
    render = True if mode == "render" else False

    # set device
    device = torch.device('cpu')

    # Set random seed
    # np.random.seed(2)
    # torch.manual_seed(cfg.seed)

    cfg.task = 'online'
    cfg.env_name = 'map6'
    cfg.env_size = '500*500'
    cfg.init_position = '50, 450'
    cfg.target_position = '450, 450'

    logging.basicConfig(level=logging.INFO, filemode='w')
    logger = logging.getLogger(__name__)

    print_line(logger, 'test')

    tester = Tester(cfg, logger, device)
    tester.load_data(model_path)
    reward = tester.run(render=render, save_path=image_save_path)
    logger.info("Reward: {:.2f}".format(reward))

    # create gif

    if mode == "fig_saving":
        png_files = os.listdir(image_save_path)
        images_list = [os.path.join(image_save_path, '{}.png'.format(i)) for i in range(len(png_files))]
        create_gif(images_list, os.path.join(save_path, "{}_{}.gif".format(cfg.env_name, episode.split("_")[0])), image_save_path)


if __name__ == "__main__":
    time_stamp = 'map7_1'
    episode = '16000_avg_reward_84.0'
    mode = "render"  # "fig_saving"
    main()