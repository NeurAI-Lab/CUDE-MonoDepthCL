from math import inf
import torch
import horovod.torch as hvd
from CUDE.trainers.base_trainer import sample_to_cuda
from CUDE.trainers_continual.er_horovod_trainer import ER_HorovodTrainer
from CUDE.utils.config import prep_logger_and_checkpoint
from CUDE.utils.depth import inv2depth, post_process_inv_depth, compute_depth_metrics
from CUDE.utils.image import flip_lr
from CUDE.utils.logging import print_config
from CUDE.utils.continual import extract_buffer_batch
from CUDE.models.EMAModel import EMAModel
from numpy import cumsum
from copy import deepcopy
from collections import OrderedDict


class MonodepthCL_HorovodTrainer(ER_HorovodTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ema_models = ['stable']
        self.logger = kwargs['logger']

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

        # MEAN-ER ADDENDA
        # EMA MODEL
        for ema_model in self.ema_models:
            for network in [attr for attr in dir(module) if "_net" in attr and ema_model not in attr]:
                setattr(module, f'{ema_model}_{network}', deepcopy(getattr(module, network)))
        self.consistency_module = EMAModel(module, self.ema_models, ema_strategy='monodepthcl')

        # set hyper-parameters for ema model
        self.reg_weight = module.config.datasets.train.continual.reg_weight
        self.begin_reg_epoch = inf if module.config.datasets.train.continual.begin_reg_epoch < 0 \
            else module.config.datasets.train.continual.begin_reg_epoch
        self.stable_update_freq = module.config.datasets.train.continual.stable_update_freq
        self.stable_alpha = module.config.datasets.train.continual.stable_alpha
        self.num_training_steps = 0

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
                # Validation for ema models
                for ema_model in self.ema_models:
                    validation_output = self.validate_ema_models(val_dataloaders, module, ema_model)
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
        # CL regularization or not
        cl_reg = True if module.current_epoch >= self.begin_reg_epoch else False
        # Start training loop
        outputs = []
        # For all batches
        for i, batch in progress_bar:
            # Reset optimizer
            depth_optimizer.zero_grad()
            pose_optimizer.zero_grad()

            # sample to be sent to buffer
            not_aug_input = {k: deepcopy(batch[k]) for k in
                             ['idx',
                              'dataset_idx',
                              'sensor_name',
                              'splitname',
                              'filename',
                              'rgb_original',
                              'intrinsics',
                              'rgb_context_original']}
            not_aug_input['rgb'] = not_aug_input.pop('rgb_original')
            not_aug_input['rgb_context'] = not_aug_input.pop('rgb_context_original')
            len_buffer_inputs = 0

            # Send samples to GPU and take a training step
            if not self.buffer.is_empty():
                buffer_inputs = self.buffer.get_data(size=self.rehearsal_batch_size,
                                                     transform=dataloader.dataset.data_transform)
                len_buffer_inputs = len(buffer_inputs)
                # CONSISTENCY LOSS
                assert all([len(batch.keys()) == len(buffer) for buffer in buffer_inputs]), f"Illformed buffer"
                # concatenate buffer_inputs with batch
                for buf_id in range(len_buffer_inputs):
                    for key in batch.keys():
                        val = buffer_inputs[buf_id][key]
                        batch[key] = self.join(batch[key], val)

            batch = sample_to_cuda(batch)
            output = module.training_step(batch, i)
            if len_buffer_inputs != 0:
                consistency_loss = self.consistency_module(extract_buffer_batch(batch, len_buffer_inputs), cl_reg)
                output["metrics"]["consistency_loss"] = consistency_loss
                output["loss"] = output["loss"] + self.reg_weight * consistency_loss
            # Backprop through loss and take an optimizer step
            output['loss'].backward()
            depth_optimizer.step()
            pose_optimizer.step()
            self.buffer.add_data(data=not_aug_input)

            self.num_training_steps += 1
            if torch.rand(1) < self.stable_update_freq:
                self.update_ema_models_variables(module)

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

    @torch.no_grad()
    def validate_ema_models(self, dataloaders, module, ema_model):
        # Set module to eval
        module.eval()
        # Start validation loop
        all_outputs = []
        print(f"\n***********Evaluating {ema_model} model***********")
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
                depth_output = self.eval_depth(module, ema_model, batch, i, task_idx=n)
                if self.logger:
                    self.logger.log_depth('val_' + ema_model, batch, depth_output, (i, n),
                                          module.validation_dataset, self.world_size,
                                          module.config.datasets.validation)
                # Append output to list of outputs
                outputs.append({'idx': batch['idx'], **depth_output['metrics']})

            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)

        # Return all outputs for epoch end
        return module.validation_epoch_end(all_outputs, name=f"{ema_model}_")

    def eval_depth(self, module, ema_model, batch, i, task_idx):
        inv_depth = getattr(module, f'{ema_model}_depth_net')(batch['rgb'])
        depth = inv2depth(inv_depth)
        inv_depth_flipped = getattr(module, f'{ema_model}_depth_net')(flip_lr(batch['rgb']))
        inv_depth_pp = post_process_inv_depth(
            inv_depth, inv_depth_flipped, method='mean')
        depth_pp = inv2depth(inv_depth_pp)
        # Calculate predicted metrics
        metrics = OrderedDict()
        if 'depth' in batch:
            for mode in module.metrics_modes:
                metrics[f"{ema_model}_" + module.metrics_name + mode] = compute_depth_metrics(
                    module.config.model.params,
                    task_idx=task_idx,
                    gt=batch['depth'],
                    pred=depth_pp if 'pp' in mode else depth,
                    use_gt_scale='gt' in mode)

        return {
            'metrics': metrics,
            'inv_depth': inv_depth_pp
        }

    def update_ema_models_variables(self, module):
        alpha = min(1 - 1 / (self.num_training_steps + 1), self.stable_alpha)
        for ema_model in self.ema_models:
            for network in [attr for attr in dir(module) if "_net" in attr and ema_model not in attr]:
                for ema_param, param in zip(getattr(module, f'{ema_model}_{network}').parameters(),
                                            getattr(module, network).parameters()):
                    ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
