# Decisions Log

## 2026-08-24: Shift from task-based to outcome-based approach

**Decision:** Stop following PLAN.md tasks sequentially. Instead, work toward the 8 concrete outcomes in the updated "Definition of done" in CLAUDE.md.

**Reasoning:** The new definition of done provides verifiable, user-facing outcomes:
1. Clean install works
2. `throttle demo` runs (<5min, no GPU, shows cost comparison)
3. `throttle cost` and `throttle tune` work against local endpoint
4. `throttle validate-sim` exists and runs
5. Error handling is clear (no tracebacks, no silent hangs)
6. Help text is plain language, jargon-free
7. QUICKSTART.md exists and works verbatim
8. CI green

**What's being skipped from PLAN.md:**
- Task 8 (500 case suite) - replaced by requirement 8 (CI green with existing tests)
- Any PLAN.md tasks not needed for the 8 outcomes will be skipped

**What's being prioritized:**
- Simulator implementation (for `throttle demo`)
- `throttle cost`, `throttle tune`, `throttle validate-sim` commands
- QUICKSTART.md
- User-facing polish (error messages, help text)

**Alternative considered:** Continue with PLAN.md tasks 4-10 sequentially.

**Why this is better:** The new definition of done is concrete, verifiable, and user-focused. It ensures a working end-to-end experience rather than completing tasks that may not contribute to usability.

This aligns with CLAUDE.md line 161-163: "If a task in PLAN.md is blocking one of them, do the work needed. If something in PLAN.md is not needed for any of the 8, skip it and note that in DECISIONS.md."
