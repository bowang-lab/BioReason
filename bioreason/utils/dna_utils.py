from typing import List, Union

import numpy as np

from transformers.utils import is_torch_available

if is_torch_available():
    import torch

DNAInput = Union[
    str, List[int], np.ndarray, "torch.Tensor", List[str], List[List[int]], List[np.ndarray], List["torch.Tensor"]
]  # noqa