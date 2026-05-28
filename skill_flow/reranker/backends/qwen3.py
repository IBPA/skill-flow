"""Qwen3-Reranker causal-LM backend (yes-token scoring).

Qwen3-Reranker is not a sequence-classification cross-encoder; it is an
instruction-following causal LM that judges relevance by emitting "yes" or
"no". The relevance score is the probability mass the model places on the
"yes" token at the final position. This class exposes a ``predict`` method
compatible with sentence-transformers' :class:`CrossEncoder` so it is a
drop-in backend for :class:`skill_flow.reranker.reranker.Reranker`. System
prompt and per-task instruction are stored as versioned text files under
``skill_flow/reranker/instructions/`` to match the project convention.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_INSTRUCTIONS_DIR = Path(__file__).resolve().parent.parent / "instructions"
DEFAULT_SYSTEM_PROMPT_PATH = _INSTRUCTIONS_DIR / "qwen3_system_v1.txt"
DEFAULT_INSTRUCTION_PATH = _INSTRUCTIONS_DIR / "qwen3_skill_v1.txt"
# Pin to the Qwen3-Reranker-0.6B commit verified against this backend.
# Override via the ``revision`` ctor arg when using other Qwen3-Reranker sizes.
DEFAULT_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"

_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


class Qwen3Reranker:
    """Scores ``(query, document)`` pairs with Qwen3-Reranker via yes-logits."""

    def __init__(
        self,
        model_name: str,
        device: str,
        *,
        revision: str = DEFAULT_REVISION,
        instruction_path: str | Path = DEFAULT_INSTRUCTION_PATH,
        system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH,
        max_length: int = 1024,
    ) -> None:
        self._device = device
        self._instruction = _read(instruction_path)
        system = _read(system_prompt_path)
        prefix = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
        # revision is required: defaults to DEFAULT_REVISION (the validated
        # Qwen3-Reranker-0.6B SHA); callers must override for other variants.
        self._tok = AutoTokenizer.from_pretrained(  # nosec B615
            model_name,
            revision=revision,
            padding_side="left",
        )
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        dtype = torch.float16 if "cuda" in str(device) else torch.float32
        model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(  # nosec B615
            model_name,
            revision=revision,
            dtype=dtype,
        )
        model = model.to(device)
        model.eval()
        self._model = model
        self._true_id = self._tok.convert_tokens_to_ids("yes")
        self._false_id = self._tok.convert_tokens_to_ids("no")
        self._prefix_ids = self._tok.encode(prefix, add_special_tokens=False)
        self._suffix_ids = self._tok.encode(_SUFFIX, add_special_tokens=False)
        self._content_max = max_length - len(self._prefix_ids) - len(self._suffix_ids)

    def _format(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self._instruction}\n<Query>: {query}\n<Document>: {doc}"

    @torch.no_grad()
    def predict(self, pairs: list[list[str]], batch_size: int = 16) -> np.ndarray:
        """Return P(relevant) for each ``[query, document]`` pair."""
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            enc = self._tok(
                [self._format(q, d) for q, d in batch],
                add_special_tokens=False,
                truncation=True,
                max_length=self._content_max,
            )
            input_ids = [
                self._prefix_ids + ids + self._suffix_ids for ids in enc["input_ids"]
            ]
            padded = self._tok.pad({"input_ids": input_ids}, return_tensors="pt").to(
                self._device,
            )
            logits = self._model(**padded).logits[:, -1, :]
            pair_logits = torch.stack(
                [logits[:, self._false_id], logits[:, self._true_id]],
                dim=1,
            )
            probs = torch.nn.functional.log_softmax(pair_logits, dim=1)[:, 1].exp()
            scores.extend(probs.float().cpu().tolist())
        return np.array(scores, dtype=np.float32)
