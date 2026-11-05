.PHONY: install dev build test test-engine test-new-engine test-api test-recovery bench holdout run clean

install:
	python3 -m pip install -r backend/requirements.txt
	cd frontend && npm install

build:
	cd frontend && npm run build

dev:
	@echo "Two terminals:"
	@echo "  1)  python3 -m uvicorn app.main:app --app-dir backend --reload"
	@echo "  2)  cd frontend && npm run dev"

run:
	./run.sh

test: test-engine test-new-engine test-api test-recovery
	cd frontend && npx tsc --noEmit

test-engine:
	python3 tests/test_engine.py

test-new-engine:
	python3 tests/test_replacement_engine.py

test-api:
	python3 tests/test_api_e2e.py

test-recovery:
	python3 tests/test_recovery.py

bench:
	python3 tests/benchmark.py

holdout:
	python3 tests/holdout.py

clean:
	rm -rf frontend/dist frontend/node_modules
	find . -name __pycache__ -type d -exec rm -rf {} +
