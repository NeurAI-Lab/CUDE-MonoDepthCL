"""
Continual Trainers
========

Continual Trainer classes providing an easy way to train and evaluate Continual SfM models
when wrapped in a ModelWrapper.

Inspired by pytorch-lightning.

"""

from CUDE.trainers_continual.er_horovod_trainer import ER_HorovodTrainer
from CUDE.trainers_continual.monodepthcl_horovod_trainer import MonodepthCL_HorovodTrainer

__all__ = ["MonodepthCL_HorovodTrainer", "ER_HorovodTrainer"]