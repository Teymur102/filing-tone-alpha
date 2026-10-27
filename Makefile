PY ?= python3

.PHONY: all setup test

all: .venv
	.venv/bin/python run.py

.venv:
	$(PY) -m venv .venv
	.venv/bin/pip install -r requirements.txt

setup: .venv

test: .venv
	.venv/bin/python -m pytest -q
