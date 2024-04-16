import math
from dataclasses import dataclass
from typing import Optional, Tuple

from fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch_nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear
)

from torch import nn

@dataclass
class ModelArgs:
    pass

class RMSNorm(torch.nn.Module):
    pass

class precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    pass

class reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    pass

class apply_rotray_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass

class repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    pass


class Attention(nn.Module):
    pass

class FeedForward(nn.Module):
    pass

class TransformerBlock(nn.Module):
    pass    

class Transformer(nn.Module):
    pass
