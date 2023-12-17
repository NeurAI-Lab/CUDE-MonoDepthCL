# Adapted from Mammoth repository for continual learning

from typing import Tuple
import numpy as np
from CUDE.trainers.base_trainer import sample_to_cpu
from CUDE.utils.types import is_tensor, is_list, is_numpy

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_seen_examples = 0
        self.attributes = ['examples']

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_dicts(self, examples) -> None:
        """
        Initializes just the examples dicts.
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = type(attr)
                setattr(self, attr_str, [typ] * self.buffer_size)

    def add_data(self, data):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_dicts(data)

        for i in range(len(data['idx'])):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                # self.examples[index] = {k: data[k][i] for k in data.keys()}
                self.examples[index] = {}
                for k in data.keys():
                    if is_tensor(data[k]):
                        self.examples[index][k] = data[k][i]
                    elif is_list(data[k]):
                        if any([is_tensor(val) for val in data[k]]):
                            # The tensor has the batch size
                            self.examples[index][k] = []
                            for val in data[k]:
                                self.examples[index][k].append(val[i])
                        else:
                            self.examples[index][k] = data[k][i]
                    else:
                        NotImplementedError()



    def get_data(self, size: int, transform=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, len(self.examples)):
            size = min(self.num_seen_examples, len(self.examples))

        choice = np.random.choice(min(self.num_seen_examples, len(self.examples)),
                                  size=size, replace=False)
        if transform is None:
            def transform(x): return x

        ret_list = tuple(
            transform({k: sample_to_cpu(ee[k])
                       for k in ee.keys()})
            for ee in [self.examples[i]
                       for i in choice])

        return ret_list

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False