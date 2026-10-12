.PHONY: help install dev test lint format clean run docker-build docker-run db-init benchmark-report benchmark-capture benchmark-seed benchmark-bakeoff benchmark-real-eval benchmark-real-eval-all deploy-edge verify-edge audit prod-ready prod-ready-operational

HOST ?= 0.0.0.0
PORT ?= 8000
BENCHMARK_ITERATIONS ?= 6
BENCHMARK_CAPTURE_INTERVAL ?= 120
BENCHMARK_PAIR_LIMIT ?= 20
BENCHMARK_POINTS_PER_PAIR ?= 16
BENCHMARK_STEP_MINUTES ?= 30
PAGES_PROJECT ?= firewatch
# Your Pages deployment, e.g. https://<project>.pages.dev; required by verify-edge and edge probes.
EDGE_BASE_URL ?=
AUDIT_EDGE_PROBE ?= 0
AUDIT_REQUIRE_MODEL_UPLIFT ?= 0
AUDIT_MIN_UPLIFT_PCT ?= 1.0
AUDIT_REQUIRE_MIN_R2 ?= 0
AUDIT_MIN_BEST_R2 ?= 0.05

help:
	@echo "Firewatch Development Commands"
	@echo "=============================="
	@echo "make install      - Install dependencies"
	@echo "make dev          - Install development dependencies"
	@echo "make run          - Run development server"
	@echo "make test         - Run tests with coverage"
	@echo "make lint         - Run linting (ruff, mypy)"
	@echo "make format       - Format code (ruff)"
	@echo "make clean        - Clean up temporary files"
	@echo "make db-init      - Initialize database"
	@echo "make benchmark-report  - Run labeled dataset quality + model benchmark report"
	@echo "make benchmark-capture - Capture live snapshots and then benchmark"
	@echo "make benchmark-seed    - Seed synthetic history and then benchmark"
	@echo "make benchmark-bakeoff - Run formal model bakeoff (incl optional boosters)"
	@echo "make benchmark-real-eval - Benchmark/tune one real labeled dataset (UCI forest-fires default)"
	@echo "make benchmark-real-eval-all - Benchmark all supported real labeled datasets"
	@echo "make deploy-edge   - Deploy Worker + static app to Cloudflare Pages"
	@echo "make verify-edge   - Verify production edge endpoints"
	@echo "make audit         - Run production readiness audit checks"
	@echo "make prod-ready    - Strict release gate (edge health + uplift + min R2)"
	@echo "make prod-ready-operational - Operational release gate (edge health, no ML uplift gate)"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-run   - Run application in Docker"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov ruff mypy

run:
	@RUN_PORT=$(PORT); \
	while lsof -nP -iTCP:$$RUN_PORT -sTCP:LISTEN >/dev/null 2>&1; do \
		RUN_PORT=$$((RUN_PORT + 1)); \
	done; \
	echo "Starting Firewatch on port $$RUN_PORT"; \
	uvicorn app.main:app --reload --host $(HOST) --port $$RUN_PORT

test:
	pytest --cov=app --cov-report=html --cov-report=term-missing tests/

lint:
	ruff check app tests audit.py
	mypy app --ignore-missing-imports

format:
	ruff check --fix app tests audit.py
	ruff format app tests audit.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist build *.egg-info

db-init:
	python -c "from app.core.database import init_db; init_db(); print('Database initialized')"

benchmark-report:
	python -m app.ml.benchmark_cli report

benchmark-capture:
	python -m app.ml.benchmark_cli capture --iterations $(BENCHMARK_ITERATIONS) --interval-seconds $(BENCHMARK_CAPTURE_INTERVAL) --run-benchmark

benchmark-seed:
	python -m app.ml.benchmark_cli seed --pair-limit $(BENCHMARK_PAIR_LIMIT) --points-per-pair $(BENCHMARK_POINTS_PER_PAIR) --step-minutes $(BENCHMARK_STEP_MINUTES) --run-benchmark

benchmark-bakeoff:
	python -m app.ml.benchmark_cli bakeoff

benchmark-real-eval:
	python -m app.ml.benchmark_cli real-eval --download-if-missing --run-tuning

benchmark-real-eval-all:
	python -m app.ml.benchmark_cli real-eval --dataset-name all --download-if-missing

deploy-edge:
	wrangler pages deploy app/static --project-name $(PAGES_PROJECT)

verify-edge:
	@test -n "$(EDGE_BASE_URL)" || { echo "Set EDGE_BASE_URL"; exit 2; }
	curl -fsS $(EDGE_BASE_URL)/edge/v1/health > /dev/null
	curl -fsS $(EDGE_BASE_URL)/docs > /dev/null
	curl -fsS $(EDGE_BASE_URL)/openapi.json > /dev/null
	@echo "Edge verification passed for $(EDGE_BASE_URL)"

audit:
	@AUDIT_ARGS=""; \
	if [ "$(AUDIT_EDGE_PROBE)" = "1" ]; then \
		test -n "$(EDGE_BASE_URL)" || { echo "Set EDGE_BASE_URL"; exit 2; }; \
		AUDIT_ARGS="$$AUDIT_ARGS --edge-base-url $(EDGE_BASE_URL)"; \
	fi; \
	if [ "$(AUDIT_REQUIRE_MODEL_UPLIFT)" = "1" ]; then \
		AUDIT_ARGS="$$AUDIT_ARGS --require-model-uplift --min-uplift-pct $(AUDIT_MIN_UPLIFT_PCT)"; \
	fi; \
	if [ "$(AUDIT_REQUIRE_MIN_R2)" = "1" ]; then \
		AUDIT_ARGS="$$AUDIT_ARGS --require-min-r2 --min-best-r2 $(AUDIT_MIN_BEST_R2)"; \
	fi; \
	python3 audit.py $$AUDIT_ARGS

prod-ready:
	$(MAKE) audit AUDIT_EDGE_PROBE=1 AUDIT_REQUIRE_MODEL_UPLIFT=1 AUDIT_REQUIRE_MIN_R2=1
	$(MAKE) verify-edge

prod-ready-operational:
	$(MAKE) audit AUDIT_EDGE_PROBE=1
	$(MAKE) verify-edge

docker-build:
	docker build -t firewatch:latest .

docker-run:
	docker-compose up

docker-run-bg:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f api
