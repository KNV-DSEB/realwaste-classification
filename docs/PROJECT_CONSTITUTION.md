# PROJECT_CONSTITUTION

These rules override convenience and last-minute experimentation.

## C1 — One source of truth
`PROJECT_SPEC.md` defines the project. No model, split, metric, or preprocessing change is valid until documented there and in `DECISION_LOG.md`.

## C2 — Same data protocol for all core models
E1, E2, and E3 must use the exact same split IDs, image size, class mapping, and evaluation code unless a documented experiment explicitly changes one factor.

## C3 — Test set is sacred
The test set is not used for architecture selection, hyperparameter tuning, augmentation choice, early stopping, or threshold selection.

## C4 — No invented evidence
No person or AI may invent accuracy, F1, class counts, training time, architecture behavior, or literature claims. Report only observed outputs or clearly labeled planned/expected behavior.

## C5 — Training-only augmentation
Random augmentation is applied only to training data. Validation and test preprocessing must be deterministic.

## C6 — Freeze dates
- End Day 7: CODE FREEZE.
- End Day 8: RESULT FREEZE.
After freeze, changes require a real bug or validity issue, not a desire for nicer numbers.

## C7 — Fair comparison
Keep input resolution, split, metric implementation, and result format comparable. Record model parameter count, best epoch, and training time when feasible.

## C8 — Scope protection
Core deliverables are E1 SimpleCNN, E2 MultiScaleCNN, E3 EfficientNet-B0. E4 augmentation ablation is optional. Grad-CAM, web demo, ensembles, extra backbones, Transformers, detection, and segmentation are stretch goals only.

## C9 — Reproducibility
Record seed, dataset snapshot/path, split file, package versions, hyperparameters, checkpoint name, and experiment ID.

## C10 — Team/AI boundary
AI can propose, review, debug, and draft. The team must run the code and is the source of truth for actual experiment outputs.
