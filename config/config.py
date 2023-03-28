# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : config.py
# Time       ：2023/3/12 上午3:37
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

import multiprocessing
from dataclasses import dataclass
import numpy as np
from dataclasses_json import dataclass_json

num_cpu = multiprocessing.cpu_count()


@dataclass_json
@dataclass
class Config:
    """ Global Parameters """
    save_path: str = ''  # save path
    model_path: str = ''
    map_path: str = './envs/maps'
    pretrained_path: str = ""  # load from pretrained file
    seed: int = 9527  # random seed
    task: str = 'online'  # coll_avoid, offline, online
    num_processes: int = 8  # how many training processes to use
    # assert num_processes <= num_cpu, "The number can't be greater than {}".format(num_cpu)

    """ Environment Parameters """
    map_type: str = 'from_image'  # 'from_image', 'human'
    env_name: str = 'test4'   # environment to train on
    render: bool = False
    robot_size: float = 5  # 0.5
    env_size: str = '500*500'
    init_position: str = '20, :'
    target_position: str = '480, :'
    init_velocity: str = ':, :'
    shape_fixed: bool = True
    turning_angle_num: int = 5
    turning_range: float = np.pi/6  # [-turning_range, turning_range]
    laser_range: float = np.pi/2  # [-laser_range, laser_range]
    laser_length: float = 100
    laser_num: int = 15
    arrive_reward_weight: float = 0.5
    collision_reward_weight: float = 0.05
    time_step_reward_weight: float = 0.01
    max_steps: int = 500
    use_proper_time_limits: bool = True

    """ Training Parameters """
    use_cuda: bool = True  # use gpu
    lr: float = 1e-5  # learning rate
    entropy_coef: float = 0.01  # entropy term coefficient
    value_loss_coef: float = 0.5  # value loss coefficient
    gamma: float = 0.99  # discount factor for rewards
    max_frames: int = 1e6
    num_steps: int = 5  # number of forward steps in A2C

    initial_interval: int = 1
    save_interval: int = 2000  # save interval, one save per n updates
    eval_interval: int = 2000  # eval interval, one eval per n updates
    eval_num: int = 10
    log_interval: int = 200  # log interval, one log per n updates

    """ Network Parameters """
    noise: bool = False
    sigma_init: float = 0.017
    recurrent: bool = False
    hidden_size: int = 256
