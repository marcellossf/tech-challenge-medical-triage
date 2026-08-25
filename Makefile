.PHONY: install lint test train export benchmark api

install:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest --cov=medical_triage --cov-report=term-missing

train:
	uv run python training/train.py

export:
	uv run python training/export_onnx.py

benchmark:
	uv run python training/benchmark.py

api:
	uv run uvicorn medical_triage.api:app --host 0.0.0.0 --port 8000 --reload

