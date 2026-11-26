.PHONY: help setup test lint format run-api docs-check

help:
	@echo "Available commands:"
	@echo "  make setup       Install core, train, and app dependencies"
	@echo "  make test        Run unit test suite with pytest"
	@echo "  make lint        Run ruff lint checks"
	@echo "  make format      Run ruff code formatting"
	@echo "  make run-api     Run FastAPI + Gradio server"
	@echo "  make docs-check  Verify documentation rules and required section headers"

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements-core.txt -r requirements-app.txt

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check .

format:
	python -m ruff format .

run-api:
	python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

docs-check:
	python scripts/check_docs.py
