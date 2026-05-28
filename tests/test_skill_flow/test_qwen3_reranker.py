"""Tests for the Qwen3-Reranker causal-LM backend (no model download)."""

from __future__ import annotations

from typing import Any, cast

import torch
from skill_flow.reranker.backends import qwen3
from skill_flow.reranker.backends.qwen3 import Qwen3Reranker


class _Padded(dict):
    """Stand-in for a tokenizer ``BatchEncoding`` with a no-op ``.to``."""

    def to(self, _device: str) -> _Padded:
        return self


class _FakeTok:
    pad_token: str | None = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, texts, **_kwargs):
        return {"input_ids": [[7, 8] for _ in texts]}

    def pad(self, batch, return_tensors=None):
        ids = batch["input_ids"]
        maxlen = max(len(x) for x in ids)
        input_ids = torch.tensor([[0] * (maxlen - len(x)) + x for x in ids])
        attn = torch.tensor([[0] * (maxlen - len(x)) + [1] * len(x) for x in ids])
        return _Padded(input_ids=input_ids, attention_mask=attn)

    def convert_tokens_to_ids(self, token: str) -> int:
        return 1 if token == "yes" else 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [100] if "system" in text else [200]


class _Out:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeModel:
    """Vocab=4; final-position logit favors 'yes' (id 1) only for sample 0."""

    def to(self, _device: str) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        b, seqlen = input_ids.shape
        logits = torch.zeros(b, seqlen, 4)
        for i in range(b):
            logits[i, -1, 1] = 5.0 if i == 0 else 0.0
            logits[i, -1, 0] = 0.0 if i == 0 else 5.0
        return _Out(logits)


def _make() -> Qwen3Reranker:
    r: Any = object.__new__(Qwen3Reranker)
    r._device = "cpu"
    r._instruction = "inst"
    r._tok = _FakeTok()
    r._model = _FakeModel()
    r._true_id = 1
    r._false_id = 0
    r._prefix_ids = [100]
    r._suffix_ids = [200]
    r._content_max = 50
    return cast("Qwen3Reranker", r)


def test_format_includes_instruction_query_and_doc() -> None:
    s = _make()._format("my query", "my doc")
    assert "<Instruct>: inst" in s
    assert "<Query>: my query" in s
    assert "<Document>: my doc" in s


def test_predict_scores_relevant_above_irrelevant() -> None:
    scores = _make().predict([["q", "rel"], ["q", "irrel"]], batch_size=2)
    assert scores.shape == (2,)
    assert scores[0] > 0.5 > scores[1]
    assert (scores >= 0).all() and (scores <= 1).all()


def test_predict_batches_cover_all_pairs() -> None:
    scores = _make().predict([["q", str(i)] for i in range(5)], batch_size=2)
    assert scores.shape == (5,)


def test_init_loads_model_and_resolves_tokens(monkeypatch) -> None:
    fake_tok = _FakeTok()
    fake_tok.pad_token = None  # exercise the eos fallback branch
    monkeypatch.setattr(
        qwen3.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: fake_tok,
    )
    monkeypatch.setattr(
        qwen3.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: _FakeModel(),
    )
    r = Qwen3Reranker("Qwen/Qwen3-Reranker-0.6B", device="cpu", max_length=256)
    assert fake_tok.pad_token == fake_tok.eos_token
    assert r._true_id == 1
    assert r._false_id == 0
    assert r._content_max == 256 - len(r._prefix_ids) - len(r._suffix_ids)
