# GATE_CHECKLISTS

## Gate 0 — Setup
PASS only if:
- [ ] GitHub repo exists and all members can access it.
- [ ] Shared Drive exists and folder structure is agreed.
- [ ] Colab notebook mounts Drive successfully.
- [ ] GPU status can be checked.
- [ ] PROJECT_SPEC has been read by the team.

## Gate 1 — Dataset Audit
PASS only if:
- [ ] class folders measured from actual downloaded copy;
- [ ] valid image count recorded;
- [ ] corrupt image count recorded;
- [ ] modes/extensions/dimensions inspected;
- [ ] class distribution calculated;
- [ ] sample grid manually reviewed;
- [ ] duplicate/near-duplicate risk considered;
- [ ] audit outputs saved;
- [ ] no unresolved data blocker.

## Gate 2 — Split Freeze
PASS only if:
- [ ] stratified 70/15/15 split created with seed 42;
- [ ] every class represented appropriately in all splits;
- [ ] split.csv saved to shared Drive;
- [ ] class mapping saved;
- [ ] all model leads use the same split file;
- [ ] test set not used for development.

## Gate 3 — E1 Baseline
PASS only if E1 can train, validate, save/load checkpoint, predict, and evaluate using shared code.

## Gate 4 — E2 Complex CNN
PASS only if E2 runs under the same data/evaluation protocol and its architecture is documented.

## Gate 5 — E3 Transfer Learning
PASS only if frozen-head and fine-tuning phases are documented and results use the same final evaluation pipeline.

## Gate 6 — Code Freeze (End Day 7)
PASS only if E1–E3 are complete or a minimum recovery plan is approved.

## Gate 7 — Result Freeze (End Day 8)
PASS only if final tables, confusion matrices, per-class metrics, and error analysis are internally consistent.

## Gate 8 — Submission
PASS only if report claims match actual results and all required assignment sections are present.
