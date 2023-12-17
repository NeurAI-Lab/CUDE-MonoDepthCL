# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import partial
from CUDE.datasets.augmentations import resize_image, resize_sample, \
    duplicate_sample, colorjitter_sample, to_tensor_sample, kitti_crop_sample, to_pil_image
from CUDE.datasets import masked_image_modeling
from torch import tensor, is_tensor
########################################################################################################################

def train_transforms(sample, image_shape, jittering, kitti_crop=False):
    """
    Training data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if (len(image_shape) > 0) and  any([is_tensor(sample[k]) for k in sample.keys()]):
        sample = to_pil_image(sample)
    if (len(image_shape) > 0) and kitti_crop:
        sample = kitti_crop_sample(sample)
    if len(image_shape) > 0:
        sample = resize_sample(sample, image_shape)
    sample = duplicate_sample(sample)
    if len(jittering) > 0:
        sample = colorjitter_sample(sample, jittering)
    sample = to_tensor_sample(sample)

    return sample

def validation_transforms(sample, image_shape, kitti_crop=False):
    """
    Validation data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if (len(image_shape) > 0) and kitti_crop:
        sample = kitti_crop_sample(sample)
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)
        try:
            sample['rgb_context'] = [resize_image(img, image_shape) for img in sample['rgb_context']]
        except:
            pass
    sample = to_tensor_sample(sample)
    return sample

def test_transforms(sample, image_shape, kitti_crop=False):
    """
    Test data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if (len(image_shape) > 0) and kitti_crop:
        sample = kitti_crop_sample(sample)
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)
        try:
            sample['rgb_context'] = [resize_image(img, image_shape) for img in sample['rgb_context']]
        except:
            pass
    sample = to_tensor_sample(sample)
    return sample

def get_transforms(mode, image_shape, jittering, kitti_crop=False, **kwargs):
    """
    Get data augmentation transformations for each split

    Parameters
    ----------
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the data augmentation transformations
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters

    Returns
    -------
        XXX_transform: Partial function
            Data augmentation transformation for that mode
    """
    if mode == 'train':
        return partial(train_transforms,
                       image_shape=image_shape,
                       jittering=jittering,
                       kitti_crop=kitti_crop,
                       **kwargs)
    elif mode == 'validation':
        return partial(validation_transforms,
                       image_shape=image_shape,
                       kitti_crop=kitti_crop)
    elif mode == 'test':
        return partial(test_transforms,
                       image_shape=image_shape,
                       kitti_crop=kitti_crop)
    else:
        raise ValueError('Unknown mode {}'.format(mode))

########################################################################################################################

