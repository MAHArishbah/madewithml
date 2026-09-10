"""Seeds, device lookup, padding and the collate function."""
# [nb cell 1, line 17]
import numpy as np
# [nb cell 1, line 21]
import os
# [nb cell 68, lines 1-2]
import random
import torch
# [nb cell 85, lines 1-2]
from ray.train.torch import get_device as ray_get_device
from functools import lru_cache


# [nb cell 69, lines 1-10]
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    eval("setattr(torch.backends.cudnn, 'deterministic', True)")
    eval("setattr(torch.backends.cudnn, 'benchmark', False)")
    os.environ["PYTHONHASHSEED"] = str(seed)


# [nb cell 85, lines 3-10]

#Ray's get_device() inside a train worker, plain torch device on the driver."""
@lru_cache(maxsize=None)
def get_device():    
    try:
        return ray_get_device()
    except RuntimeError:  # not in a Ray Train worker (e.g. notebook driver)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# [nb cell 86, lines 1-6]
def pad_array(arr, dtype=np.int32):
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype) # 2d array of 0 of row, max len
    for i, row in enumerate(arr):
        padded_arr[i][:len(row)] = row
    return padded_arr


# [nb cell 87, lines 1-5]
def collate_fn(batch):
    batch["ids"] = pad_array(batch["ids"])
    batch["masks"] = pad_array(batch["masks"])
    dtypes = {"ids": torch.int32, "masks": torch.int32, "targets": torch.int64} #targets int64 only as loss fn  expects int64
    return {key: torch.tensor(array, dtype=dtypes[key], device=get_device()) for key,array in batch.items()}
