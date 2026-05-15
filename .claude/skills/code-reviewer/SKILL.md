---
name: code-reviewer
description: >
  DCBA code reviewer. Use this skill when the user wants to review a GitHub PR —
  researching the changes in depth, discussing findings interactively, building up
  REVIEW_REMARKS.md collaboratively, and posting a formal GitHub review with anchored
  inline comments and a summary. Trigger on: /code-reviewer, /code-reviewer <N>, "review PR",
  "review this PR", "let's review", "look at PR #N", "check this PR".
---

## What this skill does

Three phases: **research -> discuss -> publish**. Do not skip ahead — the user drives the pace.

---

## Phase 1 — Research

Given a PR number, gather everything before presenting findings.

```bash
gh pr view <N> --json title,body,author,baseRefName,headRefName,files
gh pr diff <N>
gh repo view --json nameWithOwner   # capture owner/repo for later
```

Then read the full current content of every changed file — not just the diff lines.
Surrounding context often reveals the real issue.

Also read `README.md` to understand the project's goals and architecture.

### What to focus on

**Deep learning correctness** — this is the primary lens:
- Loss formulation: correct reduction, no silent bugs (e.g. double normalisation, wrong sign)
- Gradient flow: are `.detach()` calls placed correctly? anything leaking gradients it shouldn't?
- Normalisation layers (BatchNorm, LayerNorm): correct placement, train vs eval behaviour
- Dropout: applied at the right moment in the forward pass?
- Device consistency: no accidental CPU/GPU tensor splits
- Dtype consistency: mixed-precision hazards, silent upcasting
- Numerical stability: log of zero, division by near-zero, exploding/vanishing paths
- Contrastive learning specifics: temperature, L2 normalisation before similarity, negative
  sampling correctness, positive/negative mask construction
- Checkpoint completeness: all necessary state registered as buffers or handled via hooks?
- Train/val/test hygiene: data leakage, loss computed consistently across stages?

**Architecture and modularity**:
- Is logic in the right layer? (loss computation belongs in loss modules, not wrappers or
  datasets; data loading belongs in datasets, not models)
- Duplication: same logic copy-pasted across methods or classes
- Interface contracts: does this component honour the contract expected by its callers?
- Invariant violations: does the change silently break an assumption held elsewhere?
- Abstraction level: are responsibilities clear and boundaries consistent with existing patterns?

**Project fit**:
- Does the change align with what the project is trying to do (see README)?
- Are new components consistent in style and structure with existing ones?
- Are unit tests present for new stable components (data structures, loaders, utilities)?
  Generative or experimental code is exempt.

Do **not** flag style issues that pre-commit already enforces (line length, formatting,
import order, type annotations, etc.). Focus on what the code *does*, not how it looks.

---

## Phase 2 — Discuss

Present your findings concisely — **the user's opinion on every remark is final**.
If they dismiss something, drop it without argument. If they raise something you missed,
take it seriously.

Group findings by severity:

- **Must fix** — bugs, broken contracts, silent correctness failures
- **Should fix** — architectural smells, modularity violations, missing tests for stable code
- **Consider** — suggestions, alternative approaches

Use bullet points. Do not write essays. A well-placed dry observation is fine; anything
that reads as a dig at the author is not.

As each remark is agreed upon, write it to `REVIEW_REMARKS.md`:

```markdown
# PR <N> Review Remarks

## <file_path>

### Line <N> — <short imperative title>
\```python
<the offending snippet>
\```
<What is wrong, why it matters, concrete fix. 2–5 sentences. British English.>

---
```

The user may also dictate remarks directly — write those verbatim.

---

## Phase 3 — Publish to GitHub

When the user says `REVIEW_REMARKS.md` is ready, submit a single formal GitHub review.

Build the payload from `REVIEW_REMARKS.md`:

- **Summary body**: 2–3 sentences covering the most important points. Plain prose, no headers,
  no bullet list.
- **Inline comments**: one per remark, anchored to the stated line number.
- **Event**: ask the user — `COMMENT`, `APPROVE`, or `REQUEST_CHANGES`.

```bash
gh api repos/{owner}/{repo}/pulls/{N}/reviews \
  --method POST \
  --input - <<'EOF'
{
  "body": "<summary paragraph>",
  "event": "COMMENT",
  "comments": [
    {"path": "src/foo.py", "line": 42, "side": "RIGHT", "body": "<remark text>"},
    ...
  ]
}
EOF
```

Use RIGHT side and the new-file line number for each comment. For remarks on deleted lines
with no corresponding new line, anchor to the last line of the enclosing hunk.

Print the review URL once posted.
