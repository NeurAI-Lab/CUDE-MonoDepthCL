# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image

from CUDE.utils.misc import filter_dict
########################################################################################################################
def to_pil_image(sample):
    image_transform = transforms.ToPILImage()
    for key in filter_dict(sample, [
        'rgb', 'rgb_original'
    ]):
        sample[key] = image_transform(sample[key])
    # context images
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original'
    ]):
        sample[key] = [image_transform(k) for k in sample[key]]

    return sample
########################################################################################################################

def resize_image(image, shape, interpolation=Image.ANTIALIAS):
    """
    Resizes input image.

    Parameters
    ----------
    image : Image.PIL
        Input image
    shape : tuple [H,W]
        Output shape
    interpolation : int
        Interpolation mode

    Returns
    -------
    image : Image.PIL
        Resized image
    """
    transform = transforms.Resize(shape, interpolation=interpolation)
    return transform(image)

def resize_depth(depth, shape):
    """
    Resizes depth map.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W]
        Resized depth map
    """
    depth = cv2.resize(depth, dsize=shape[::-1],
                       interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(depth, axis=2)

def resize_sample_image_and_intrinsics(sample, shape,
                                       image_interpolation=Image.ANTIALIAS):
    """
    Resizes the image and intrinsics of a sample

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and corresponding intrinsics
    image_transform = transforms.Resize(shape, interpolation=image_interpolation)
    (orig_w, orig_h) = sample['rgb'].size
    (out_h, out_w) = shape
    # Scale intrinsics
    for key in filter_dict(sample, [
        'intrinsics'
    ]):
        intrinsics = np.copy(sample[key])
        intrinsics[0] *= out_w / orig_w
        intrinsics[1] *= out_h / orig_h
        sample[key] = intrinsics
    # Scale images
    for key in filter_dict(sample, [
        'rgb', 'rgb_original'
    ]):
        sample[key] = image_transform(sample[key])
    # Scale context images
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original'
    ]):
        sample[key] = [image_transform(k) for k in sample[key]]
    # Return resized sample
    return sample

def kitti_crop_depth(depth, crop_top, crop_left, out_h, out_w):

    depth = depth[crop_top: crop_top + out_h, crop_left: crop_left + out_w]
    return depth

def kitti_crop_sample_image_and_intrinsics(sample, crop_top, crop_left, out_h, out_w):

    shape = (out_h, out_w)
    image_transform = transforms.CenterCrop(shape)

    # Change corresponding intrinsics
    for key in filter_dict(sample, [
        'intrinsics'
    ]):
        intrinsics = np.copy(sample[key])
        intrinsics[0, 2] -= crop_left
        intrinsics[1, 2] -= crop_top
        sample[key] = intrinsics

    # Center Crop images
    for key in filter_dict(sample, [
        'rgb', 'rgb_original'
    ]):
        sample[key] = image_transform(sample[key])
    # Center Crop context images
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original'
    ]):
        sample[key] = [image_transform(k) for k in sample[key]]

    return sample

def kitti_crop_sample(sample):
    """
    Crop a sample, including image, intrinsics, and gt depth maps to kitti like shape

    Parameters
    sample : dict
        Dictionary with sample values

   Returns
    -------
    sample : dict
        Resized sample

    """
    KITTI_W = 1242
    KITTI_H = 375
    KITTI_ratio = KITTI_W / KITTI_H

    (orig_w, orig_h) = sample['rgb'].size
    orig_ratio = orig_w / orig_h

    # If more than +/- 1% difference
    if abs(KITTI_ratio - orig_ratio) / KITTI_ratio > 1 / 100:
        if (KITTI_ratio > orig_ratio):
            crop_type = "height"
        else:
            crop_type = "width"
    else:
        return sample

    if crop_type == "height":
        out_w = orig_w
        out_h = int(np.round(orig_w / KITTI_ratio))
    else:  # crop_type == "width"
        out_h = orig_h
        out_w = int(round(orig_h * KITTI_ratio))

    crop_left = int((orig_w - out_w + 1) * 0.5)
    crop_top = int((orig_h - out_h + 1) * 0.5)

    sample = kitti_crop_sample_image_and_intrinsics(sample=sample,
                                                    crop_top=crop_top, crop_left=crop_left,
                                                    out_h=out_h, out_w=out_w)

    # Crop depth maps
    for key in filter_dict(sample, [
        'depth'
    ]):
        sample[key] = kitti_crop_depth(sample[key],
                                       crop_top=crop_top, crop_left=crop_left,
                                       out_h=out_h, out_w=out_w)
    # Crop depth contexts
    for key in filter_dict(sample, [
        'depth_context'
    ]):
        sample[key] = [kitti_crop_depth(k,
                                        crop_top=crop_top, crop_left=crop_left,
                                        out_h=out_h, out_w=out_w)
                       for k in sample[key]]

    # Return cropped sample
    return sample

def resize_sample(sample, shape, image_interpolation=Image.ANTIALIAS):
    """
    Resizes a sample, including image, intrinsics and depth maps.

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and intrinsics
    sample = resize_sample_image_and_intrinsics(sample, shape, image_interpolation)
    # Resize depth maps
    for key in filter_dict(sample, [
        'depth'
    ]):
        sample[key] = resize_depth(sample[key], shape)
    # Resize depth contexts
    for key in filter_dict(sample, [
        'depth_context'
    ]):
        sample[key] = [resize_depth(k, shape) for k in sample[key]]
    # Return resized sample
    return sample

########################################################################################################################

def to_tensor(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return transform(image).type(tensor_type)

def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    transform = transforms.ToTensor()
    # Convert single items
    for key in filter_dict(sample, [
        'rgb', 'rgb_original', 'depth'
    ]):
        sample[key] = transform(sample[key]).type(tensor_type)
    # Convert lists
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original', 'depth_context'
    ]):
        sample[key] = [transform(k).type(tensor_type) for k in sample[key]]
    # Return converted sample
    return sample

########################################################################################################################

def duplicate_sample(sample):
    """
    Duplicates sample images and contexts to preserve their unaugmented versions.

    Parameters
    ----------
    sample : dict
        Input sample

    Returns
    -------
    sample : dict
        Sample including [+"_original"] keys with copies of images and contexts.
    """
    # Duplicate single items
    for key in filter_dict(sample, [
        'rgb'
    ]):
        sample['{}_original'.format(key)] = sample[key].copy()
    # Duplicate lists
    for key in filter_dict(sample, [
        'rgb_context'
    ]):
        sample['{}_original'.format(key)] = [k.copy() for k in sample[key]]
    # Return duplicated sample
    return sample

def colorjitter_sample(sample, parameters, prob=1.0):
    """
    Jitters input images as data augmentation.

    Parameters
    ----------
    sample : dict
        Input sample
    parameters : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    prob : float
        Jittering probability

    Returns
    -------
    sample : dict
        Jittered sample
    """
    if random.random() < prob:
        # Prepare transformation
        brightness, contrast, saturation, hue = parameters
        color_augmentation = transforms.ColorJitter(
            brightness=[max(0, 1 - brightness), 1 + brightness],
            contrast=[max(0, 1 - contrast), 1 + contrast],
            saturation=[max(0, 1 - saturation), 1 + saturation],
            hue=[-hue, hue])
        augment_image = color_augmentation

        # Jitter single items
        for key in filter_dict(sample, [
            'rgb'
        ]):
            sample[key] = augment_image(sample[key])
        # Jitter lists
        for key in filter_dict(sample, [
            'rgb_context'
        ]):
            sample[key] = [augment_image(k) for k in sample[key]]
    # Return jittered (?) sample
    return sample

########################################################################################################################


