import torch
import torch.nn.functional as F
from CUDE.utils.types import is_tensor, is_list


def slice_buffer_batch(value, buffer_batch_size):
    if is_tensor(value):
        n_dims = len(value.shape)
        if n_dims == 1:
            return value[-buffer_batch_size:]
        elif n_dims in list(range(2, 6)):
            # 2 = mask, 3 = intrinsics, 4 = regular nchw batches, 5 = multi-cam
            return value[-buffer_batch_size:, :]
        else:
            raise NotImplementedError("Tensor should 1, 2, 3, 4, or 5 dimensions")
    elif is_list(value):
        if all(is_tensor(el) for el in value):
            buf_value = []
            for i in range(len(value)):
               buf_value.append(slice_buffer_batch(value[i], buffer_batch_size))
            return buf_value
        else:
            return value[-buffer_batch_size:]
    else:
        NotImplementedError("Unseen datatype in batch")


def extract_buffer_batch(batch, buffer_batch_size):
    if 'mask' in batch:
        for i, mask in enumerate(batch['mask']):
            if mask is None:
                batch['mask'][i] = torch.empty(len(batch['idx']), 0)
    buffer_batch = {}
    for key, value in batch.items():
        buffer_batch[key] = slice_buffer_batch(value, buffer_batch_size)
    return buffer_batch

def RKDAngle(student, teacher):
    with torch.no_grad():
        td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = (student.unsqueeze(0) - student.unsqueeze(1))
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')

    return loss