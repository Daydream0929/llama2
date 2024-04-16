import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama2.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role,
    content: str

class CompletionPrediction(TypedDict, total=False):
    generation: str,
    tokens: List[str]
    logprobs: List[float]

class ChatPrediction(TypedDict, total=False):
    generation: Message,
    tokens: List[str]
    logprobs: List[float]

Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

class Llama2:
    pass

def sample_top_p(probs, p):
    pass

