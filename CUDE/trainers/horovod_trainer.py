# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
import horovod.torch as hvd
from CUDE.trainers.base_trainer import BaseTrainer, sample_to_cuda
from CUDE.utils.config import prep_logger_and_checkpoint
from CUDE.utils.logging import print_config, pcolor
from CUDE.utils.logging import AvgMeter
from numpy import cumsum
from CUDE.models.model_wrapper import print0

class HorovodTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        hvd.init()
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
        torch.cuda.set_device(hvd.local_rank())
        torch.backends.cudnn.benchmark = True

        self.avg_loss = AvgMeter(50)
        self.dtype = kwargs.get("dtype", None)  # just for test for now

    @property
    def proc_rank(self):
        return hvd.rank()

    @property
    def world_size(self):
        return hvd.size()

    def print_task(self, task_num, total_tasks):

        def wrap(string):
            return '| {} |'.format(string)

        hor_line = '|{:<}|'.format('*' * 93)
        print()
        print(hor_line)
        task_info = wrap(pcolor('***** {:>33} / {:<43} *****'.format("Task " + str(task_num + 1), total_tasks), 'yellow', attrs=['bold']))
        print0(task_info)
        print(hor_line)
        print()

    def fit(self, module):

        # Prepare module for training
        module.trainer = self
        # Update and print module configuration
        prep_logger_and_checkpoint(module)
        print_config(module.config)

        # Send module to GPU
        module = module.to('cuda')
        # Configure optimizer and scheduler
        module.configure_optimizers()

        # Create distributed optimizer
        compression = hvd.Compression.none
        depth_optimizer = hvd.DistributedOptimizer(module.depth_optimizer,
                                             named_parameters=module.named_parameters(), compression=compression)
        pose_optimizer = hvd.DistributedOptimizer(module.pose_optimizer,
                                                   named_parameters=module.named_parameters(), compression=compression)
        depth_scheduler = module.depth_scheduler
        pose_scheduler = module.pose_scheduler

        # Get train and val dataloaders
        train_dataloaders = module.train_dataloaders()
        val_dataloaders = module.val_dataloader()

        assert len(train_dataloaders) == len(self.max_epochs), \
            f"The number of tasks ({len(train_dataloaders)}) and " \
            f"the list of max epochs ({len(self.max_epochs)}) should have equal lengths for training."

        # Task loop
        for n, [train_dataloader_task, max_epochs_cumulative] in enumerate(zip(train_dataloaders, cumsum(self.max_epochs))):
            module.current_task = n

            self.print_task(n, len(train_dataloaders))
            # Epoch loop
            for epoch in range(module.current_epoch, max_epochs_cumulative):
                # Train
                self.train(train_dataloader_task, module, depth_optimizer, pose_optimizer)
                # Validation
                validation_output = self.validate(val_dataloaders, module)
                # Check and save model
                self.check_and_save(module, validation_output)
                # Update current epoch
                module.current_epoch += 1
                # Take a scheduler step
                depth_scheduler.step()
                pose_scheduler.step()

    def train(self, dataloader, module, depth_optimizer, pose_optimizer):
        # Set module to train
        module.train()
        print(module.logs['depth_learning_rate'], module.logs['pose_learning_rate'])
        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)
        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        # Start training loop
        outputs = []
        # For all batches
        for i, batch in progress_bar:
            # Reset optimizer
            depth_optimizer.zero_grad()
            pose_optimizer.zero_grad()
            # Send samples to GPU and take a training step
            batch = sample_to_cuda(batch)
            output = module.training_step(batch, i)
            # Backprop through loss and take an optimizer step
            output['loss'].backward()
            depth_optimizer.step()
            pose_optimizer.step()
            # Append output to list of outputs
            output['loss'] = output['loss'].detach()
            outputs.append(output)
            # Update progress bar if in rank 0
            if self.is_rank_0:
                progress_bar.set_description(
                    'Epoch {}/{} | Avg.Loss {:.4f}'.format(
                        module.current_epoch + 1, sum(self.max_epochs), self.avg_loss(output['loss'].item())))
        # Return outputs for epoch end
        return module.training_epoch_end(outputs)

    def validate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start validation loop
        all_outputs = []
        # For all validation datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.validation, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a validation step
                batch = sample_to_cuda(batch)
                output = module.validation_step(batch, i, n, task_idx=n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.validation_epoch_end(all_outputs)

    def test(self, module):
        # Send module to GPU
        module = module.to('cuda', dtype=self.dtype)
        # Get test dataloaders
        test_dataloaders = module.test_dataloader()
        # Run evaluation
        output = self.evaluate(test_dataloaders, module)
        return output

    @torch.no_grad()
    def evaluate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start evaluation loop
        all_outputs = []
        # For all test datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.test, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a test step
                batch = sample_to_cuda(batch, self.dtype)
                output = module.test_step(batch, i, n, task_idx=n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.test_epoch_end(all_outputs)
