import os
from logging import getLogger
from typing import list

from sentencepiece import SentencePieceProcessor

logger = getLogger()

class Tokenizer:
    """tokenizing and encoding/decoding text using SentencePiece"""

    def __init__(self, model_path: str):
        pass

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        pass

    def decode(self, t: List[int]) -> str:
        pass
