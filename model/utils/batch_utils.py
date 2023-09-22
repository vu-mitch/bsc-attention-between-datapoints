import warnings
from collections import OrderedDict, defaultdict

import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import column_or_1d

import torch
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

collate_with_pre_batching_err_msg_format = (
    "collate_with_pre_batched_map: "
    "batch must be a list with one map element; found {}")


def collate_with_pre_batching(batch):
    r"""
    Collate function used by our PyTorch dataloader (in both distributed and
    serial settings).

    We avoid adding a batch dimension, as for NPT we have pre-batched data,
    where each element of the dataset is a map.

    :arg batch: List[Dict] (not as general as the default collate fn)
    """
    if len(batch) > 1:
        raise NotImplementedError

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, container_abcs.Mapping):
        return elem  # Just return the dict, as there will only be one in NPT

    raise TypeError(collate_with_pre_batching_err_msg_format.format(elem_type))


# TODO: batching over features?
