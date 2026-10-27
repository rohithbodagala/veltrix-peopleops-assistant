PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: install test run up down docker-test smoke

install:  ## create .venv with runtime + test dependencies
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

test:  ## run the test suite (synthetic corpus + DB; ./data is optional)
	$(PYTHON) -m pytest -q -W ignore::DeprecationWarning

run:  ## run the API locally on :8000 using .env
	set -a; [ -f .env ] && . ./.env; set +a; $(PYTHON) -m uvicorn app.main:create_app --factory --port 8000

up:  ## build and start the service with docker compose
	docker compose up --build

down:
	docker compose down

docker-test:  ## run the test suite inside the image (tesseract available there)
	docker compose run --rm --no-deps -v $(CURDIR)/data:/app/data assistant python -m pytest -q -W ignore::DeprecationWarning

smoke:  ## scripted conversation against a running server (meaningful with a real LLM key)
	$(PYTHON) scripts/smoke_live.py
