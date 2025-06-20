---
description: 
globs: 
alwaysApply: false
---
---
title: Mini-ChatGPT Engineering Rulebook
schema: "codestyle/v1"
description: Unified guidelines for coding, testing, documentation, and release.
globs:
  alwaysApply: true
---
# Mini ChatGPT Coding Guidelines
 
 While some existing code may not strictly follow these standards (due to historical reasons), we are gradually refactoring the codebase to align with these guidelines.

# Tooling & Environment
* **Python** ≥ 3.11; use `conda` for isolation.
* **Environment** use `conda activate sesame` before using commands to activate environment
* **Package Manager**: `pip` + `requirements/*.txt`
* **Editable install**:  
  ```bash
  pip install -e .

# Coding Standards
* **Formatter** `black` · **linter** `ruff` · **import sort** `isort`.  
  Run all with `pre-commit run -a`.  
* **Typing**: `mypy --strict`; warnings are errors.  
* Follow PEP 8 & the five key Zen lines: *Beautiful · Explicit · Simple · Readable · Practical*.  
* Functions/classes should fit on one screen; extract helpers otherwise.  
* Prefer composition over inheritance.

# Testing Strategy
* **pytest ≥ 8** with `--import-mode=importlib`.  
* Layout mirrors source tree (`tests/unit/model/test_forward.py`).  
* **Coverage** ≥ 90 % (`pytest --cov=mini_chat_gpt`).  
* Fixtures go in `tests/conftest.py`; no `sys.path` tweaks.  
* Test edge cases, error conditions, and every public API surface.

# Pre-commit & Continuous Integration
* Hooks: `black`, `ruff`, `mypy`, `pytest`, `detect-secrets`.  
* Enable **pre-commit.ci** so hooks run on every PR.  
* GitHub Actions workflow: lint → test → build → upload coverage to Codecov.


# Documentation
* **Docstring style**: Google.  
* Required for every public symbol.  
* Build docs with `mkdocs-material`; auto-deploy on GitHub Pages.

# Performance & Profiling
* Use `torch.profiler` or `cProfile` before each release.  
* Enable `torch.compile` + AMP when GPU available; fall back gracefully.  
* Track throughput & memory in `wandb`.

# Error Handling & Logging
* Assertions for developer errors; raise typed exceptions for user errors.  
* Log with `structlog` (`info`, `warning`, `error`); never `print`.  
* Do **not** swallow exceptions — re-raise with context if needed.

# Agent Automation
Implement / modify functionality
Generate / update tests
Run pytest
while tests fail:
    Refactor code and/or tests
    Run pytest


