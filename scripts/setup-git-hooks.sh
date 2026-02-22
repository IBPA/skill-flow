#!/bin/bash

uv run pre-commit install && uv run pre-commit install --hook-type pre-push
