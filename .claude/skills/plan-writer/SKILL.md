---
name: plan-writer
description: >
  Creates a PLAN.md implementation plan for tasks in the DCBA ecosystem. Use this skill whenever
  the user asks to write up a plan, plan out a feature, prepare a plan for an implementation, or
  says things like "let's plan X", "write a plan for X", "how should we approach implementing Y",
  or "plan this out". Produces a concise, step-by-step PLAN.md that the dev-assistant can then
  execute directly.
---

## Purpose

Produce a focused, actionable `PLAN.md` in the repository root that a developer (or the
dev-assistant skill) can follow step by step without further research.

Good plans are short: a handful of clearly-ordered steps, each with enough context to understand
*why* it is needed and enough specificity (concrete commands, file paths, expected outcomes) to
execute it without guesswork.

The user owns the design — your job is to ask good questions, capture their thinking, and
translate it into a concrete, executable plan. Do not make architectural or design decisions
independently. If you find yourself researching "what the best approach would be", stop and ask
the user instead.

---

## Workflow

### 1. Interview the user

Before touching the codebase, have a short conversation to understand what the user wants to
build and how. Ask focused questions — one or two at a time — to surface:

- What the feature does and why it is needed
- Key design decisions: interfaces, data structures, algorithms, dependencies
- Any constraints or preferences they have (e.g. "keep it simple", "mirror the existing X pattern")

Keep going until you have enough to write a concrete plan. A good signal: you could hand the
plan to someone else and they would know exactly what to build without having to ask the user
anything.

Do not ask for information you can look up yourself (file paths, existing class names, etc.).

### 2. Look up only what is needed to be concrete

Once you know *what* to build, do targeted codebase lookups to fill in the specifics the plan
needs: exact file paths, class and function names to extend or call, config keys, etc.

Use direct Glob/Grep/Read calls for this — it should be quick and focused, not a broad
exploration.

### 3. Write PLAN.md

Write the plan to `PLAN.md` in the repository root (overwrite any existing one unless the user
explicitly asks to append or keep the old plan).

---

## PLAN.md format

```markdown
# <Short descriptive title>

## Steps

### 0. <Step name>

<One or two sentences explaining what this step does and why it is needed.>

<Concrete commands, config snippets, or code fragments — enough to execute without further
research. Use fenced code blocks.>

<Optional: how to verify this step succeeded.>

### 1. <Step name>

...
```

Rules:
- Steps are numbered from 0
- Keep step count low — prefer 4–7 steps; split only when two things are genuinely independent
- Each step should be completable in one sitting
- Commands must be copy-pasteable and match the DCBA stack (`uv`, `git`, `dvc`, `pre-commit`)
- Do not include steps the dev-assistant already performs automatically (e.g., running pre-commit
  before every commit — that is already in its workflow)
- Write in British English throughout
- Be concrete: name actual files, modules, and functions rather than "update the relevant file"

---

## Example step (for reference)

```markdown
### 2. Add `GraphLoader` to the public API

`dcba/graph_io.py` currently exports `load_report` but not `GraphLoader`. Add it to `__all__`
so downstream code can import it directly from the package.

Edit `dcba/__init__.py`:
```python
from dcba.graph_io import GraphLoader, load_report
```

Verify: `uv run python -c "from dcba import GraphLoader; print(GraphLoader)"` should print the
class without error.
```

---

## After writing

Tell the user that `PLAN.md` is ready and give a one-sentence summary of the approach. Do not
reproduce the entire plan in the chat — they can read the file.
