"""Conftest for src tests — mocks heavy ML dependencies before collection."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

# sentence_transformers pulls in multiprocess, which has a syntax error on
# Python 3.14.  Pre-inject a stub so that ``from sentence_transformers import
# SentenceTransformer`` succeeds at collection time without loading the real
# package.
_st_stub = ModuleType("sentence_transformers")
_st_stub.SentenceTransformer = MagicMock  # type: ignore[attr-defined]
_st_stub.CrossEncoder = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("sentence_transformers", _st_stub)
