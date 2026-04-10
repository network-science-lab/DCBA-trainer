---
name: dev-assistant
description: >
  Development assistant for DCBA ecosystem repositories (Python + uv stack). Use this skill whenever
  working on any development task in a DCBA repo — writing code, debugging, running experiments,
  adding features, managing dependencies, handling data with DVC, or committing changes. Provides
  coding conventions, active workflow guidance, and project standards.
---

## Session start

At the start of every session, read `README.md` in the working directory root to get repo-specific
context: purpose, setup instructions, usage, and available commands.

---

## Language

Use **British English** in all text: comments, docstrings, commit messages, and documentation.

---

## Code style

- Line length: **100 characters**
- **Type hints** are required on every function signature (arguments and return type)
- **Docstrings**: reStructuredText style, Sphinx field directives
  - Use `:param x:` and `:returns:` only — omit `:type x:`, `:rtype:`, and `:raises:`
  - Class attribute documentation: plain prose, no `:ivar:`
  - One-liner docstrings stay on a single line
  - Multiline docstrings: text on a new line after `"""`, with `:returns:` separated from `:param`
    fields by a blank line
- **File paths**: always `pathlib.Path`, never bare strings
- **File I/O**: always specify `encoding="utf-8"` explicitly on every `open()` / `Path.open()`

**Docstring examples:**

```python
# One-liner
def retry_count(attempts: int) -> bool:
    """Return True if the number of attempts has not exceeded the limit."""


# Multiline
def generate_graph(config: GraphConfig, output_dir: Path) -> Path:
    """
    Generate a synthetic graph from the given configuration and write it to disk.

    Runs the appropriate Julia-backed generator based on ``config.experiment_type``
    and writes edge and community files to ``output_dir``.

    :param config: Generation parameters including degree distribution and community sizes.
    :param output_dir: Directory where edge and community files will be written.

    :returns: Path to the directory containing the generated output files.
    """
```

---

## Available CLI tools

| Tool         | Purpose                                                                            |
| ------------ | ---------------------------------------------------------------------------------- |
| `uv`         | Python environment and package management; running scripts and tests               |
| `git`        | Version control                                                                    |
| `dvc`        | Data versioning — tracking, pushing, and pulling datasets                          |
| `pre-commit` | Code quality hooks — invoke via `uv run pre-commit`                                |
| `juliapkg`   | Julia dependency resolution — invoke via `uv run python -c "import juliapkg; ..."` |

---

## Package management

Always use `uv add <pkg>` to add dependencies. Never use `pip install`.

---

## New branch workflow

Before making any other changes on a new branch:

1. Check the current `version` in `pyproject.toml` on `master`
2. Bump `version` in `pyproject.toml`
3. Then proceed with the branch work

---

## Git workflow

- Move files with `git mv`, never bare `mv`
- Do not commit known-broken code — test before committing
- Run `uv run pre-commit run --files <changed files>` before every commit
- Commit messages: short imperative subject line in British English
- No co-authorship trailers in commit messages

---

## DVC

- Track a new data directory: `dvc add <path>`
- Push data to remote: `dvc push`
- Pull data: `dvc pull`

---

## Validation and data modelling

Use **Pydantic v2** at system boundaries: config files, CLI input, and external data (files, API
responses). For internal data, prefer plain dataclasses or typed dicts.

Never use bare `assert` for input validation — `assert` is stripped by Python's `-O` flag. Use
Pydantic validators or explicit `if`/`raise` instead. Reserve `assert` for tests only.

- Cross-field validation: `@model_validator`
- Single-field validation: `@field_validator`
- Configuration: `model_config = ConfigDict(...)`, not inner `class Config`

---

## Unit testing

When adding or modifying stable components (data structures, loaders, utilities), propose unit tests
alongside the implementation. Experimental or generative code (graph generators, Julia-backed
algorithms) is exempt.

---

## Security

- Do not hardcode secrets, tokens, or passwords in source files — use environment variables or a
  secrets manager
- Do not log sensitive values (credentials, personal data, raw user input)
- `rm -rf` and `git push --force` / `git push -f` are blocked at the harness level
