import numpy as np
import torch


def get_numpy_dtype(dtype_name):
    if dtype_name == 'float32':
        dtype = np.float32
    elif dtype_name == 'float64':
        dtype = np.float64
    else:
        raise NotImplementedError

    return dtype


def get_torch_dtype(dtype_name):
    if dtype_name == 'float32':
        dtype = torch.float32
    elif dtype_name == 'float64':
        dtype = torch.float64
    else:
        raise NotImplementedError

    return dtype


def get_torch_tensor_type(dtype_name):
    if dtype_name == 'float32':
        dtype = torch.FloatTensor
    elif dtype_name == 'float64':
        dtype = torch.DoubleTensor
    else:
        raise NotImplementedError

    return dtype


def torch_cast_to_dtype(obj, dtype_name):
    if dtype_name == 'float32':
        obj = obj.float()
    elif dtype_name == 'float64':
        obj = obj.double()
    elif dtype_name == 'long':
        obj = obj.long()
    else:
        raise NotImplementedError

    return obj

