# claude.md

## Environment Requirement

All commands, scripts, and executions MUST be run inside the `ml_env` conda environment.

---

## Setup Instructions

Before running any code, always activate the environment:

```bash
conda activate ml_env
```

If activation is not persistent (e.g., in scripts or CI), use:

```bash
conda run -n ml_env <command>
```

---

## Execution Rules

* Never run Python, pip, or shell commands outside of `ml_env`
* Always assume dependencies are installed in `ml_env`
* If a command fails, first verify that the environment is active
* Include environment activation in any generated scripts or instructions

---

## Guidance for Claude

* Default to using:

```bash
conda run -n ml_env python <script>.py
```

* When writing commands, always ensure they run within `ml_env`
* Do not assume the base environment is acceptable

---

## Strict Enforcement

Do NOT execute any command unless `ml_env` is active or explicitly specified.

If uncertain, activate the environment before proceeding.
