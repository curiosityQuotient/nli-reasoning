# AGENTS.md - Project Context & Constraints

> **CRITICAL FOR AGENTS:** Read this file before proposing any architectural changes, writing code, or installing dependencies. Do not deviate from these engineering standards without explicit user approval.

---

## 1. Project Overview & Architecture
* **Project Name:** [Your Project Name]
* **Description:** [One or two sentences explaining what this application actually does.]
* **Architecture Style:** Modular, decoupled, and highly typed. Avoid massive monolithic files. Prefer single-responsibility modules under a clear domain layout.

### Directory Structure Blueprint
```text
├── src/               # Core code that set the logic
│   ├── utils/         # Utilities that are needed across multiple core functions
├── tests/             # Mirroring the src/ layout for unit and integration tests
├── AGENTS.md          # This file (Agent Memory)
└── pyproject.toml     # Dependency and tool configurations
