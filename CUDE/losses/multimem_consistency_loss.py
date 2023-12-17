# Adapted from PackNet Toyota Research Institute.  All rights reserved.

import torch

from CUDE.utils.image import match_scales
from CUDE.geometry.camera import Camera
from CUDE.geometry.camera_utils import view_synthesis
from CUDE.utils.depth import inv2depth
from CUDE.losses.loss_base import LossBase, ProgressiveScaling
from CUDE.losses.multiview_photometric_loss import SSIM
from CUDE.utils.continual import RKDAngle
from random import gauss, randint
########################################################################################################################


class MultiMemConsistencyLoss(LossBase):
    """
    Self-Supervised multiview photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them

    Parameters
    ----------
    num_scales : int
        Number of inverse depth map scalesto consider
    ssim_loss_weight : float
        Weight for the SSIM loss
    occ_reg_weight : float
        Weight for the occlusion regularization loss
    smooth_loss_weight : float
        Weight for the smoothness loss
    C1,C2 : float
        SSIM parameters
    photometric_reduce_op : str
        Method to reduce the photometric loss
    disp_norm : bool
        True if inverse depth is normalized for
    clip_loss : float
        Threshold for photometric loss clipping
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    padding_mode : str
        Padding mode for view synthesis
    automask_loss : bool
        True if automasking is enabled for the photometric loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, config, model_names, ema_strategy='monodepthcl'):
        super().__init__()
        required_attributes = [
            "num_scales",
            "ssim_loss_weight",
            "occ_reg_weight",
            "C1", "C2",
            "progressive_scaling",
            "padding_mode",
            "automask_loss",
            "multimem_image_consistency_crop",
            "multimem_image_consistency_loss_weight",
            "multimem_depth_consistency_loss_weight"
        ]
        for attr in required_attributes:
            setattr(self, attr, getattr(config, attr))

        self.progressive_scaling = ProgressiveScaling(
            self.progressive_scaling, self.num_scales)
        self.prediction_names = model_names
        self.ema_strategy = ema_strategy

    def warp_ref_image(self, inv_depths, ref_image, K, ref_K, pose):
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(self.num_scales):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(Camera(K=K.float()).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.num_scales)]
        ref_images = match_scales(ref_image, inv_depths, self.num_scales)
        ref_warped = [view_synthesis(
            ref_images[i], depths[i], ref_cams[i], cams[i],
            padding_mode=self.padding_mode) for i in range(self.num_scales)]
        # Return warped reference image
        return ref_warped

########################################################################################################################

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural SIMilarity) loss

        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        kernel_size : int
            Convolutional parameter

        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM loss
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def get_gaussian_crop(self, orig_height, orig_width):
        MEAN = 0.5  # 50%
        STD = 0.1  # 10%
        multiplier_size = gauss(mu=MEAN, sigma=STD)

        # Clip multiplier if it is too small
        if multiplier_size < 0.1:
            multiplier_size = 0.1
        elif multiplier_size > 1.0:
            multiplier_size = 1.0

        new_height = int(round(orig_height * multiplier_size))
        new_width = int(round(orig_width * multiplier_size))
        top = randint(0, orig_height - new_height)
        left = randint(0, orig_width - new_width)
        return top, left, new_height, new_width

    def crop_losses(self, losses):
        for i, loss in enumerate(losses):
            # TODO: finalize if different or same crops for each scale
            orig_height, orig_width = list(loss.shape[-2:])
            top, left, new_height, new_width = self.get_gaussian_crop(orig_height, orig_width)
            losses[i] = loss[...,
                        top: top + new_height,
                        left: left + new_width]
        return losses

    def image_consistency_loss(self, refs_warped):
        # L1 loss
        num_contexts = len(refs_warped['working_model'])
        consistency_loss = 0.0
        for context in range(num_contexts):
            l1_loss = [torch.abs(refs_warped['working_model'][context][i] - refs_warped['stable'][context][i])
                       for i in range(self.num_scales)]
            # SSIM loss
            if self.ssim_loss_weight > 0.0:
                ssim_loss = [self.SSIM(
                    refs_warped['working_model'][context][i],
                    refs_warped['stable'][context][i],
                    kernel_size=3) for i in range(self.num_scales)]
                # Weighted Sum: alpha * ssim + (1 - alpha) * l1
                losses = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                          (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                          for i in range(self.num_scales)]
            else:
                losses = l1_loss
            # Only return losses on cropped regions
            if self.multimem_image_consistency_crop:
                losses = self.crop_losses(losses)
            consistency_loss += sum([l.mean() for l in losses]) / len(losses)
        return consistency_loss / num_contexts

    def scale_inv_depth(self, inv_depth):
        min_inv_depth = torch.min(inv_depth)
        max_inv_depth = torch.max(inv_depth)
        return (inv_depth - min_inv_depth)/(max_inv_depth - min_inv_depth)

    def depth_consistency_loss(self, inv_depths):
        scaled_inv_depths = {}
        losses = []
        for i in range(self.num_scales):
            scaled_inv_depths['working_model'] = self.scale_inv_depth(inv_depths['working_model'][i])
            scaled_inv_depths['stable'] = self.scale_inv_depth(inv_depths['stable'][i])
            losses.append(torch.abs(scaled_inv_depths['working_model'] - scaled_inv_depths['stable']))
        consistency_loss = sum([l.mean() for l in losses]) / len(losses)
        return consistency_loss

    def monodepthcl_consistency_loss(self, refs_warped, inv_depths):
        """

        Parameters
        ----------
        refs_warped : A dictionary of lists with reconstructions made from working model and stable model.
        Each model generates a list with reconstructions from each "context" image. Each context image
        leads to reconstructions at 4 scales.

        Returns
        -------
        A reconstruction consistency loss averaged across all pixels, scales, and contexts.
        """
        loss = 0.0
        # TODO: MAKE THE CONSISTENCY LOSS FUNCTIONS GENERALIZE TO MULTIPLE EMA MODELS
        if self.multimem_image_consistency_loss_weight > 0:
            loss += self.multimem_image_consistency_loss_weight * self.image_consistency_loss(refs_warped)
        if self.multimem_depth_consistency_loss_weight > 0:
            loss += self.multimem_depth_consistency_loss_weight * self.depth_consistency_loss(inv_depths)
        return loss

    def relational_similarity_loss(self, inv_depths):
        rs_loss = 0.0
        for scale in range(self.num_scales):
            rs_loss += RKDAngle(
                torch.flatten(inv_depths['working_model'][scale], start_dim=1),  # student
                torch.flatten(inv_depths['stable'][scale], start_dim=1)  # teacher
            )
        rs_loss /= self.num_scales
        return rs_loss

    def forward(self, contexts, inv_depths,
                K, poses, cl_reg=False, progress=0.0):
        """
        Calculates training photometric loss.

        Parameters
        ----------
        context : list of torch.Tensor [B,3,H,W]
            Context containing a list of reference images
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        poses : list of Pose
            Camera transformation between original and context
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # If using progressive scaling
        self.num_scales = self.progressive_scaling(progress)
        # Loop over all reference images
        refs_warped = {}
        for prediction_name in self.prediction_names:
            refs_warped[prediction_name] = []
            for j, (ref_image, pose) in enumerate(zip(contexts, poses[prediction_name])):
                # Calculate warped images
                if isinstance(K, dict):
                    ref_warped = self.warp_ref_image(inv_depths[prediction_name], ref_image,
                                                     K[prediction_name], K[prediction_name],
                                                     pose)
                else:
                    ref_warped = self.warp_ref_image(inv_depths[prediction_name], ref_image, K, K, pose)
                refs_warped[prediction_name].append(ref_warped)

        #Calculate and store image loss
        if self.ema_strategy == "monodepthcl":
            loss = self.monodepthcl_consistency_loss(refs_warped, inv_depths) if cl_reg else 0.0
        else:
            NotImplementedError("Strategy not implemented.")

        # Return loss
        return loss
########################################################################################################################
