# Drone Forest — a 3D simulator whose flight decisions come from a pluggable engine.
SHELL := /bin/bash
.DEFAULT_GOAL := help
UV := uv
RUN := $(UV) run
PY := $(RUN) python
export HF_HUB_OFFLINE ?= 1
export TOKENIZERS_PARALLELISM ?= false

SERVICE_URL := http://127.0.0.1:8765
GAME_URL := http://127.0.0.1:5173
SERVER_LOG := /tmp/df-server.log
SEED ?= 7
SECONDS ?= 45

.PHONY: finetune eval-offline collect probe help setup serve web dev test lint build arena telemetry clean

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_. -]+:[^=]*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":[^#]*?## "} {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Install the Python service (uv) and the web game (npm)
	$(UV) sync --group dev
	cd web && npm install

serve: ## Run the decision microservice on :8765 (DEFAULT_ENGINE=laya|heuristic|random)
	@$(RUN) uvicorn server.app:app --host 127.0.0.1 --port 8765

web: ## Run the game dev server on :5173
	cd web && npm run dev

dev: ## Service in the background (log: /tmp/df-server.log) + game in the foreground; Ctrl-C stops both
	@$(RUN) uvicorn server.app:app --host 127.0.0.1 --port 8765 > $(SERVER_LOG) 2>&1 & \
	  pid=$$!; trap "kill $$pid 2>/dev/null" EXIT INT TERM; \
	  for i in $$(seq 1 60); do curl -sf $(SERVICE_URL)/health > /dev/null && break; sleep 0.5; done; \
	  echo "service  $(SERVICE_URL)/health   (engines: $$(curl -sf $(SERVICE_URL)/health | $(PY) -c 'import json,sys; print(\", \".join(json.load(sys.stdin)[\"engines\"]))' 2>/dev/null || echo 'not up yet, see $(SERVER_LOG)'))"; \
	  echo "game     $(GAME_URL)/?engine=heuristic&seed=$(SEED)"; \
	  echo "         $(GAME_URL)/?engine=laya&seed=$(SEED)"; \
	  echo "arena    make arena  (prints the URLs)"; \
	  cd web && npm run dev

test: ## Python tests (Laya tests need the cached checkpoint; they skip otherwise)
	@$(RUN) pytest -q

lint: ## ruff + tsc
	@$(RUN) ruff check server tests
	cd web && npm run typecheck

build: ## Production build of the game into web/dist
	cd web && npm run build

arena: ## Print the URL-driven benchmark runs (same seed for every engine); needs `make dev` running
	@echo "Open each URL, wait for the ARENA panel to say done, then read the ARENA_RESULT line in the console:"
	@for e in heuristic laya random local; do \
	  echo "  $(GAME_URL)/?arena=quality&seconds=$(SECONDS)&engine=$$e&seed=$(SEED)"; done
	@echo "  ...&arena=realtime  keeps the wall clock, so a slow engine loses decision slots exactly as in play"
	@echo "Or programmatically: window.__droneForest.arenaDone.then(console.log)"

telemetry: ## Summarise the newest telemetry file in data/ (FILE=path for another one)
	@$(PY) -m server.telemetry $(FILE)

clean: ## Remove build output, caches and telemetry files (keeps .venv and node_modules)
	rm -rf web/dist web/.vite .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -not -path './web/node_modules/*' -exec rm -rf {} +
	rm -f data/telemetry-*.jsonl

probe: ## Ask the running service to decide one frame: tree 6 m dead ahead, left clear 60 m (ENGINE=laya|heuristic|random)
	@for i in 1 2 3; do curl -s -X POST http://127.0.0.1:8765/decide -H 'content-type: application/json' \
	  -d "$$(python3 -c "import json;d=json.load(open('scripts/tree_ahead_frame.json'));d['engine']='$(or $(ENGINE),laya)';print(json.dumps(d))")" \
	  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  {d['engine']:<10} {d['action']:<11} P={d['probabilities'][d['action']]:.3f}  conf={d.get('confidence') or 0:.2f}  think={d['latency_ms']:.1f} ms  -- {d['reason']}\")"; done

finetune: ## Fine-tune Laya on recorded telemetry with the heuristic as teacher -> data/checkpoints/laya-drone-ft (EPOCHS=8)
	@PYTHONUNBUFFERED=1 $(PY) -m server.finetune --epochs $(or $(EPOCHS),8) --batch 12 --balance 1 --half-encoder 1 $(FINETUNE_ARGS)

eval-offline: ## Score question formulations / a checkpoint against the teacher on recorded frames (CHECKPOINT=english|data/checkpoints/laya-drone-ft)
	@$(PY) -m server.offline_eval --per-class $(or $(PER_CLASS),60) --checkpoint $(or $(CHECKPOINT),english) --formulations $(or $(FORMULATIONS),choice_semantic)

collect: ## Print the URL that records a batch of teacher-labelled runs into data/ (needs `make serve` + `make web`)
	@echo "Open in a browser (16 chained arena runs, ~12 min, ~7,000 labelled frames):"
	@echo "  http://127.0.0.1:5173/?arena=quality&seconds=45&engine=heuristic&seed=100&seeds=100-115"
	@echo "Telemetry lands in data/telemetry-*.jsonl; then: make finetune"
