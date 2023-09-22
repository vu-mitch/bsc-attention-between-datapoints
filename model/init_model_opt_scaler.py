import pprint

from torch import optim
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from model.npt import NPTModel
from model.utils.encode_utils import get_torch_dtype

from model.utils.optim_utils import Lookahead, Lamb


def init_model_opt_scaler(c, metadata, device=None):
    if device is None:
        device = c.exp_device

    model = NPTModel(
        c, metadata=metadata, device=device)

    model_torch_dtype = get_torch_dtype(dtype_name=c.model_dtype)
    model = model.to(device=device).type(model_torch_dtype)
    print(f'Model has {count_parameters(model)} parameters,'
          f'batch size {c.exp_batch_size}.')

    optimizer = init_optimizer(
        c=c, model_parameters=model.parameters(), device=device)
    print(f'Initialized "{c.exp_optimizer}" optimizer.')

    # Automatic Mixed Precision (AMP)
    # If c.model_amp is False, the GradScaler call becomes a no-op
    # so we can switch between default/mixed precision without if/else
    # statements.
    scaler = GradScaler(enabled=c.model_amp)
    if c.model_amp:
        print(f'Initialized gradient scaler for Automatic Mixed Precision.')

    return model, optimizer, scaler


def init_optimizer(c, model_parameters, device):
    if 'default' in c.exp_optimizer:
        optimizer = optim.Adam(params=model_parameters, lr=c.exp_lr)
    elif 'lamb' in c.exp_optimizer:
        lamb = Lamb
        optimizer = lamb(
            model_parameters, lr=c.exp_lr, betas=(0.9, 0.999),
            weight_decay=c.exp_weight_decay, eps=1e-6)
    else:
        raise NotImplementedError

    if c.exp_optimizer.startswith('lookahead_'):
        optimizer = Lookahead(optimizer, k=c.exp_lookahead_update_cadence)

    return optimizer


def get_sorted_params(model):
    param_count_and_name = []
    for n,p in model.named_parameters():
        if p.requires_grad:
            param_count_and_name.append((p.numel(), n))

    pprint.pprint(sorted(param_count_and_name, reverse=True))


def count_parameters(model):
    r"""
    Due to Federico Baldassarre
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
