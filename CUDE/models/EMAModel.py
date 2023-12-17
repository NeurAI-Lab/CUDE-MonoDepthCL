import random
import torch.nn as nn

from CUDE.utils.image import flip_model, interpolate_scales
from CUDE.geometry.pose import Pose
from CUDE.geometry.camera_utils import construct_K_from_predictions
from CUDE.utils.misc import make_list
from CUDE.losses.multimem_consistency_loss import MultiMemConsistencyLoss


class EMAModel(nn.Module):
    """
    Model that inherits a depth and pose network from SfmModel and
    includes the photometric loss for self-supervised training.

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """
    def __init__(self,
                 module, ema_model_names, ema_strategy='monodepthcl'):
        super().__init__()

        # depth ema models
        self.depth_models = {}
        # ASSUMPTION: ONLY ONE LEARNING MODEL
        self.depth_models['working_model'] = module.depth_net
        for ema_model in ema_model_names:
            self.depth_models[ema_model] = getattr(module, f'{ema_model}_depth_net')

        # pose ema models
        self.pose_models = {}
        # ASSUMPTION: ONLY ONE LEARNING MODEL
        self.pose_models['working_model'] = module.pose_net
        for ema_model in ema_model_names:
            self.pose_models[ema_model] = getattr(module, f'{ema_model}_pose_net')

        # hyper-params
        self.flip_lr_prob = module.config.model.loss.flip_lr_prob
        self.rotation_mode = module.config.model.loss.rotation_mode
        self.learn_intrinsics = module.config.model.pose_net.learn_intrinsics
        self.upsample_depth_maps = module.config.model.loss.upsample_depth_maps
        self.ema_strategy = ema_strategy

        # Initializes the consistency loss
        self.ema_consistency_loss = MultiMemConsistencyLoss(module.config.model.loss,
                                                            ['working_model'] + ema_model_names,
                                                            ema_strategy=self.ema_strategy)

    def compute_inv_depths(self, image, mask=None):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        flip_lr = random.random() < self.flip_lr_prob if self.training else False
        inv_depths = {}
        for model_name, depth_model in self.depth_models.items():
            inv_depths[model_name] = make_list(flip_model(depth_model, image, mask, flip_lr))
        # If upsampling depth maps
        if self.upsample_depth_maps:
            for model_name in inv_depths.keys():
                inv_depths[model_name] = interpolate_scales(
                    inv_depths[model_name], mode='nearest', align_corners=None)
        # Return inverse depth maps
        return inv_depths

    def compute_poses(self, image, contexts, masks=None):
        """Compute poses from image and a sequence of context images"""
        poses = {}
        if self.learn_intrinsics:
            focal_lengths = {}
            offsets_all = {}

        for model_name, pose_net in self.pose_models.items():
            if masks is not None and all([m is not None for m in masks]):
                if self.learn_intrinsics:
                    pose_vec, focal_length, offsets = pose_net(image, contexts, masks)
                else:
                    pose_vec = self.pose_net(image, contexts, masks)
            else:
                if self.learn_intrinsics:
                    pose_vec, focal_length, offsets = pose_net(image, contexts)
                else:
                    pose_vec = pose_net(image, contexts)

            pose = [Pose.from_vec(pose_vec[:, i], self.rotation_mode)
                    for i in range(pose_vec.shape[1])]
            poses[model_name] = pose
            if self.learn_intrinsics:
                focal_lengths[model_name] = focal_length
                offsets_all[model_name] = offsets

        if self.learn_intrinsics:
            return poses, focal_lengths, offsets_all
        else:
            return poses

    def forward(self, batch, cl_reg=False, progress=0.0):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored

        Returns
        -------
        output : dict
            Dictionary containing predicted inverse depth maps and poses
        """
        masks = {}
        if 'mask' in batch:
            for i, mask in enumerate(batch['mask']):
                if mask.nelement() > 0:
                    mask = mask.unsqueeze(1)
                else:
                    mask = None
                batch['mask'][i] = mask
            masks['rgb'] = batch['mask'][0]
            masks['rgb_context'] = batch['mask'][1:]
        else:
            masks['rgb'] = None
            masks['rgb_context'] = None

        # Generate inverse depth predictions
        inv_depths = self.compute_inv_depths(batch['rgb'], masks['rgb'])
        # Generate pose predictions if available
        poses = None
        K = batch['intrinsics']
        if 'rgb_context' in batch:
            if self.learn_intrinsics:
                K = {}
                poses, focal_lengths, offsets_all = self.compute_poses(batch['rgb'],
                                                                  batch['rgb_context'],
                                                                  masks['rgb_context'])
                for model_name in list(self.pose_models.keys()):
                    K[model_name] = construct_K_from_predictions(focal_lengths=focal_lengths[model_name],
                                                                 offsets=offsets_all[model_name],
                                                                 batch_size=len(batch['intrinsics']),
                                                                 img_shape=tuple(batch['rgb'].shape[-2:]))
            else:
                poses = self.compute_poses(batch['rgb'],
                                          batch['rgb_context'],
                                          masks['rgb_context'])
        consistency_loss = self.ema_consistency_loss(
            batch['rgb_context_original'],
            inv_depths, K, poses, cl_reg
        )
        return consistency_loss