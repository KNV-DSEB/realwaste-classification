# ROADMAP — 10 DAYS

## Day 1 — Environment + Dataset Audit
Outputs:
- GitHub + Drive + Colab setup
- dataset inventory
- class counts
- corruption check
- image dimension/mode check
- sample grid
- duplicate-risk notes
- Gate 1 review

## Day 2 — Split + SimpleCNN end-to-end
Outputs:
- frozen `split.csv`
- shared dataloaders
- E1 can train, validate, save, load, predict, evaluate

## Day 3 — Freeze E1 + implement E2
Outputs:
- E1 final baseline run
- E1 curves/metrics
- E2 architecture implementation

## Day 4 — Train/fix E2
Outputs:
- E2 stable run
- E2 architecture diagram
- preliminary comparison E1 vs E2

## Day 5 — E3 transfer learning
Outputs:
- EfficientNet-B0 head training
- Stage-A checkpoint
- E3 pipeline validated

## Day 6 — Fine-tune E3 + consolidate pipeline
Outputs:
- Stage-B fine-tuning
- E1/E2/E3 validation results
- report model-design sections drafted

## Day 7 — Final core experiments
Outputs:
- final E1–E3 test evaluation
- optional E4 augmentation ablation only if stable
- CODE FREEZE

## Day 8 — Analysis
Outputs:
- final model-comparison table
- per-class metrics
- confusion matrices
- 50–100 error samples or a feasible structured sample
- failure-mode analysis
- RESULT FREEZE

## Day 9 — Report completion
Outputs:
- theory
- problem/dataset
- preprocessing
- architecture diagrams
- experiment results
- discussion/conclusion
- references
- report ≥90% final

## Day 10 — QA + presentation
Outputs:
- factual consistency audit
- reproducibility check
- final report
- slides
- rehearsal
- submission archive

## Scope-drop order if behind schedule
1. Keep E1/E2/E3.
2. Keep fair evaluation.
3. Keep report.
4. Keep confusion/error analysis.
5. Drop E4.
6. Drop Grad-CAM.
7. Drop demo.
8. Never drop a lecturer-required model to preserve a stretch goal.
