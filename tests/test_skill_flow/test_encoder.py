"""Tests for src.index.encoder."""

from unittest.mock import MagicMock, patch

import numpy as np
from skill_flow.config import RetrieverConfig
from skill_flow.index.encoder import Encoder


def _make_mock_model():
    """Create a mock SentenceTransformer that returns random embeddings."""
    model = MagicMock()

    def fake_encode(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        vecs = np.random.default_rng(42).random((n, 768)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    model.encode = fake_encode
    return model


@patch.object(Encoder, "__init__", lambda self, *a, **kw: None)
def test_encode_documents_shape():
    encoder = Encoder()
    encoder._config = RetrieverConfig()
    encoder._model = _make_mock_model()

    texts = ["hello world", "foo bar", "test sentence"]
    result = encoder.encode_documents(texts)

    assert result.shape == (3, 768)
    assert result.dtype == np.float32


@patch.object(Encoder, "__init__", lambda self, *a, **kw: None)
def test_encode_documents_normalized():
    encoder = Encoder()
    encoder._config = RetrieverConfig()
    encoder._model = _make_mock_model()

    result = encoder.encode_documents(["hello", "world"])
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@patch.object(Encoder, "__init__", lambda self, *a, **kw: None)
def test_encode_query_shape():
    encoder = Encoder()
    encoder._config = RetrieverConfig()
    encoder._model = _make_mock_model()

    result = encoder.encode_query("test query")

    assert result.shape == (1, 768)
    assert result.dtype == np.float32


@patch.object(Encoder, "__init__", lambda self, *a, **kw: None)
def test_encode_query_uses_prefix():
    encoder = Encoder()
    encoder._config = RetrieverConfig()
    model = MagicMock()
    model.encode = MagicMock(return_value=np.ones((1, 768), dtype=np.float32))
    encoder._model = model

    encoder.encode_query("test query")

    call_args = model.encode.call_args
    texts = call_args[0][0]
    assert texts == [encoder._config.query_prompt + "test query"]
