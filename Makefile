# Laya lab — self-hosted decision-model bootstrap and capability scenarios.
# Every target is safe to re-run.

SHELL := /bin/bash
.DEFAULT_GOAL := help
UV := uv
RUN := $(UV) run
PY := $(RUN) python

# Scenarios run offline once `make warm` has cached the checkpoints.
export HF_HUB_OFFLINE ?= 1
export TOKENIZERS_PARALLELISM ?= false

.PHONY: help upstream setup warm doctor test test-fast verify-loader link-models clean distclean \
        lint gpu-check scenarios demo quick demo-quick verify s1 s2 s3 s4 s5 s6 \
        routing guard confidence throughput limits-language limits-footguns

help: ## Show this help
	@echo ""
	@echo "  Laya lab — self-hosted typed-decision engine"
	@echo ""
	@grep -hE '^[a-zA-Z0-9_. -]+:[^=]*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":[^#]*?## "} {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""

upstream: ## Clone/refresh the Laya library into ./upstream at the pinned commit
	@bash tools/fetch_upstream.sh

setup: upstream ## Fetch upstream, create the venv, install pinned deps (idempotent)
	$(UV) sync --frozen --group dev
	@echo "venv ready at .venv  (laya installed editable from ./upstream)"

warm: ## Download the three checkpoints into the HF cache (~2.3 GB, once)
	@HF_HUB_OFFLINE=0 $(PY) tools/warm_cache.py

lint: ## Run ruff over scenarios/ and tools/
	@$(RUN) ruff check scenarios tools

doctor: ## Verify interpreter, deps, device, checkpoints and one real forward pass
	@$(PY) tools/doctor.py

test: ## Run the full upstream Laya test suite (8 suites, incl. real-weights e2e, ~3 min)
	@$(PY) tools/run_upstream_tests.py

test-fast: ## Upstream suites that need no model weights (~5 s)
	@$(PY) tools/run_upstream_tests.py test_ --skip-e2e

gpu-check: ## Prove all three checkpoints run on the Mac GPU and agree with CPU (~60 s)
	@$(PY) tools/gpu_check.py

verify-loader: ## Prove the fast checkpoint loader is output-identical to stock laya.load()
	@$(PY) tools/verify_loader.py

link-models: ## Expose cached checkpoints as local dirs at ~/laya_models (symlinks)
	@$(PY) tools/link_local_models.py

# ---------------------------------------------------------------- scenarios
# Each scenario is a standalone script that prints a readable report, writes its raw
# numbers to out/, and exits non-zero if its core claim stops holding.

S1 := scenarios/01_routing_table.py
S2 := scenarios/02_guard_firewall.py
S3 := scenarios/03_confidence_gate.py
S4 := scenarios/04_throughput.py
S5 := scenarios/05_language_limit.py
S6 := scenarios/06_silent_footguns.py

s1 routing: ## Scenario 1 - routing is free: 19 inputs in 10 languages, zero models loaded (~10 s)
	@$(PY) $(S1)

s2 guard: ## Scenario 2 - prompt-injection firewall: 5 safety decisions in 36 ms, offline (~12 s)
	@$(PY) $(S2)

s3 confidence: ## Scenario 3 - confidence is not accuracy: 18/18 correct at mean confidence 0.59 (~5 s)
	@$(PY) $(S3)

s4 throughput: ## Scenario 4 - N questions, one forward pass: what batching actually buys (~21 s)
	@$(PY) $(S4)

s5 limits-language: ## Scenario 5 - the limit: where the router is blind, and what a mis-route costs (~13 s)
	@$(PY) $(S5)

s6 limits-footguns: ## Scenario 6 - four ways your input is destroyed with no error at all (~8 s)
	@$(PY) $(S6)

scenarios demo: ## Run all six scenarios in order (~70 s); stops at the first broken claim
	@$(PY) $(S1) && $(PY) $(S2) && $(PY) $(S3) && $(PY) $(S4) && $(PY) $(S5) && $(PY) $(S6)
	@echo ""
	@echo "  All scenarios passed. Raw numbers are in out/."

quick demo-quick: ## The one scenario that needs no model weights at all (~10 s)
	@$(PY) $(S1)

verify: doctor test scenarios ## Full proof run: doctor, upstream suite, every scenario
	@echo ""
	@echo "  doctor + upstream suite + all scenarios green."

clean: ## Remove scenario outputs and caches
	@rm -rf out/* __pycache__ scenarios/__pycache__ tools/__pycache__ .pytest_cache
	@echo "cleaned"

distclean: clean ## Also remove the venv (keeps the downloaded checkpoints)
	@rm -rf .venv
	@echo "removed .venv (HF checkpoint cache kept; delete ~/.cache/huggingface to reclaim ~2.3 GB)"
