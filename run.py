# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : run.py
# Time       ：2023/3/12 上午3:30
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

import datetime
import logging
import os
import pickle
import time
import torch

from config import config, dump_config
from model.trainer import Trainer
from model.utils import print_line


def main():
    # Set path
    now_time = datetime.datetime.now()
    now_time = datetime.datetime.strftime(now_time, "%Y_%m_%d_%H_%M_%S")
    save_path = os.getcwd()
    config.save_path = "{}/result/{}".format(save_path, now_time)
    config.model_path = os.path.join(config.save_path, 'checkpoint')
    os.makedirs(config.model_path) if not os.path.exists(config.model_path) else None
    dump_config(config.save_path, config)

    # set device
    # torch.set_num_threads(1)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')

    # Set random seed
    # np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(1)

    # Set the logger
    logging.basicConfig(level=logging.INFO, filemode='w')
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(os.path.join(config.save_path, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Device: {}".format(device))

    start = time.time()

    trainer = Trainer(config, logger, device)

    print_line(logger, 'train')

    returns = trainer.run()

    with open(os.path.join(config.save_path, 'results.pickle'), 'wb') as f:
        pickle.dump(returns, f)

    logger.info("----------------------------------------")
    logger.info("        Done. Time cost: {:.2f}h        ".format((time.time() - start) / 3600))
    logger.info("----------------------------------------")


if __name__ == "__main__":
    main()
