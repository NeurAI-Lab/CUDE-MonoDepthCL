
import torch
from CUDE.trainers.base_trainer import sample_to_cuda
from CUDE.trainers.horovod_trainer import HorovodTrainer
from CUDE.utils.types import is_tensor, is_list, is_numpy

from CUDE.utils.buffer import Buffer
from copy import deepcopy

class ER_HorovodTrainer(HorovodTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        buffer_size = kwargs.get("buffer_size", 0)
        rehearsal_batch_size = kwargs.get("rehearsal_batch_size", 0)
        self.buffer = Buffer(buffer_size=buffer_size)
        self.rehearsal_batch_size = rehearsal_batch_size

    def train(self, dataloader, module, depth_optimizer, pose_optimizer):
        # Set module to train
        module.train()
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

            # ER strategy
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

            if not self.buffer.is_empty():
                buffer_inputs = self.buffer.get_data(size=self.rehearsal_batch_size,
                                                     transform=dataloader.dataset.data_transform)
                assert all([len(batch.keys()) == len(buffer) for buffer in buffer_inputs]), f"Illformed buffer"
                # concatenate buffer_inputs with batch
                for j in range(len(buffer_inputs)):
                    for key in batch.keys():
                        val = buffer_inputs[j][key]
                        batch[key] = self.join(batch[key], val)

            # Send samples to GPU and take a training step
            batch = sample_to_cuda(batch)
            output = module.training_step(batch, i)
            # Backprop through loss and take an optimizer step
            output['loss'].backward()
            depth_optimizer.step()
            pose_optimizer.step()

            # ER strategy
            self.buffer.add_data(data=not_aug_input)

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

    def join(self, batch_in, buffer_in):
        if is_tensor(batch_in):
            batch_in = torch.cat([batch_in, torch.unsqueeze(torch.tensor(buffer_in), dim=0)])
        elif is_list(batch_in):
            if isinstance(buffer_in, list):
                assert len(batch_in) == len(buffer_in), "Illformed buffer"
                for i in range(len(batch_in)):
                    batch_in[i] = self.join(batch_in[i], buffer_in[i])
            else:
                batch_in.append(buffer_in)
        else:
            NotImplementedError()

        return batch_in