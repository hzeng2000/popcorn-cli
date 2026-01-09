import torch
from typing import Tuple

# Input type: tuple of two matrices (A: m×k, B: k×n)
input_t = Tuple[torch.Tensor, torch.Tensor]

# Output type: result matrix (C: m×n)
output_t = torch.Tensor
