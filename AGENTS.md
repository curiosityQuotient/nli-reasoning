# AGENTS.md - Project Context & Constraints

> **CRITICAL FOR AGENTS:** Read this file before proposing any architectural changes, writing code, or installing dependencies. Do not deviate from these engineering standards without explicit user approval.

---

## 1. Project Overview & Architecture
* **Project Name:** Two stage NLI learning
* **Description:** This is some code that takes two-stage-learning.ipynb and implements it in a more maintainable way in a git repo. It should replicate the functionality of the notebook, due to this it had to be run on kaggle. 
* **Architecture Style:** Modular, decoupled, and highly typed. Avoid massive monolithic files. Prefer single-responsibility modules under a clear domain layout.

### Directory Structure Blueprint
```text
├── src/               # Core code that set the logic
│   ├── utils/         # Utilities that are needed across multiple core functions
├── tests/             # Mirroring the src/ layout for unit and integration tests
├── AGENTS.md          # This file (Agent Memory)
└── pyproject.toml     # Dependency and tool configurations


## 2. Technical Stack & Tooling Constraints

* **Python Version:** `^3.11` (Leverage modern syntax: `TaskGroup`, advanced type hinting, etc.)
* **Dependency Management:** Use `uv` — do not randomly add packages via raw `pip install` without updating the configuration file.
* **Testing Framework:** `pytest` (Prefer functional tests, fixtures, and parameterized testing).
* **Linting & Formatting:** `ruff` (Strict linting rules, automated sorting of imports).

---

## 3. Python Coding Standards & Paradigms

### Type Hinting & Safety

* **Strict Typing:** All function signatures *must* include type hints for parameters and return types. Use explicit types from `typing` or built-ins.
* **No `Any`:** Avoid using `typing.Any` unless absolutely unavoidable. Define precise types, `TypeVar`, or structural protocols instead.

### Code Style & Layout

* **Function Length:** Keep functions focused. If a function exceeds 50 lines, evaluate if it should be broken down into composable utilities.
* **Async/Await:** Use `asyncio` natively for I/O-bound operations (network calls, database queries). Keep CPU-bound work isolated.
* **Error Handling:** Avoid generic `try/except Exception: pass` blocks. Catch explicit exceptions, log them accurately, and re-raise or handle gracefully.

---

## 4. Current State & Active Task

### Current Milestone

* Building out the core data processing pipeline and setting up the initial testing suite.

### Active Task for Agent

* **Focus:** [e.g., "Implement the repository pattern for the database adapter in `src/infrastructure/` and write corresponding pytest fixtures."]
* **Next Steps:**
1. Draft the abstract base class for the data storage interface.
2. Implement the local file system or memory-based concrete adapter.
3. Write a test suite ensuring 100% code coverage for the new adapter.



---

## 5. Persistent Memory / Lessons Learned


