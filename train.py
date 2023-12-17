# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse

from CUDE.models.model_wrapper import ModelWrapper
from CUDE.models.model_checkpoint import ModelCheckpoint
from CUDE.trainers.horovod_trainer import HorovodTrainer
from CUDE.trainers_continual.er_horovod_trainer import ER_HorovodTrainer
from CUDE.trainers_continual.monodepthcl_horovod_trainer import MonodepthCL_HorovodTrainer
from CUDE.utils.config import parse_train_file
from CUDE.utils.load import set_debug, filter_args_create
from CUDE.utils.horovod import hvd_init, rank
from CUDE.loggers import WandbLogger

from warnings import warn
import os

def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM training script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), \
        'You need to provide a .ckpt of .yaml file'
    return args


def train(file):
    """
    Monocular depth estimation training script.

    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    # Initialize horovod
    hvd_init()

    # Produce configuration and checkpoint from filename
    config, ckpt = parse_train_file(file)

    # Set debug if requested
    set_debug(config.debug)

    if config.logger.type == 'wandb':
        # Wandb Logger
        logger = None if config.wandb.dry_run or rank() > 0 \
            else filter_args_create(WandbLogger, config.wandb)
    else:
        logger = None
        warn("Logger invalid. Not logging.")

    # model checkpoint
    checkpoint = None if config.checkpoint.filepath is '' or rank() > 0 else \
        filter_args_create(ModelCheckpoint, config.checkpoint)

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config, resume=ckpt, logger=logger)

    # Create trainer with args.arch parameters
    CONTINUAl_STRATEGIES = ['', 'er', 'monodepthcl']
    if config.datasets.train.continual.enabled:
        assert config.datasets.train.continual.strategy in CONTINUAl_STRATEGIES, "Please specify cl strategy from those available"

    if config.datasets.train.continual.strategy.strip() == "er":
        trainer = ER_HorovodTrainer(**config.arch, checkpoint=checkpoint,
                                    rehearsal_batch_size=config.datasets.train.continual.rehearsal_batch_size,
                                    buffer_size=config.datasets.train.continual.buffer_size)
    elif config.datasets.train.continual.strategy.strip() == "monodepthcl":
        trainer = MonodepthCL_HorovodTrainer(**config.arch, checkpoint=checkpoint,
                                             rehearsal_batch_size=config.datasets.train.continual.rehearsal_batch_size,
                                             buffer_size=config.datasets.train.continual.buffer_size,
                                             logger=logger)
    else:
        trainer = HorovodTrainer(**config.arch, checkpoint=checkpoint)

    # Train model
    trainer.fit(model_wrapper)


if __name__ == '__main__':
    args = parse_args()
    train(args.file)
