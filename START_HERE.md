# RealWaste Deep Learning Project — Starter Pack v1.0

## Purpose
This pack is the operating system for a 10-day Deep Learning group project on RealWaste. It is designed to keep the team, ChatGPT, and Claude aligned around one fixed experimental protocol and the lecturer's required deliverables.

## Locked project
**Title:** Robust Real-World Waste Classification using Custom CNN Architectures and Transfer Learning

**Task:** Multi-class image classification on the user-selected RealWaste dataset.

**Core models:**
1. SimpleCNN — custom CNN from scratch.
2. MultiScaleCNN — custom complex CNN from scratch.
3. EfficientNet-B0 — ImageNet transfer learning and fine-tuning.

**Core metrics:** Accuracy, Macro Precision, Macro Recall, Macro F1, per-class metrics, confusion matrix.

**Core split:** One fixed stratified Train/Validation/Test split = 70/15/15. Seed = 42.

## Use this pack in this exact order
1. Read `docs/PROJECT_CONTEXT.md`.
2. Read `docs/PROJECT_CONSTITUTION.md`.
3. Read `docs/PROJECT_SPEC.md`.
4. Read `docs/RESEARCH_STATE.md`.
5. Read `docs/TEAM_OPERATING_SYSTEM.md`.
6. Create one private GitHub repository and copy this pack into it.
7. Create one shared Google Drive project folder for dataset/checkpoints/results/report.
8. Open `notebooks/00_setup.ipynb` in Google Colab.
9. Run `notebooks/01_dataset_audit.ipynb`.
10. Stop at Gate 1 and send the audit outputs to ChatGPT for PASS/FIX review.

## Single Source of Truth
- **Project scope/protocol:** `docs/PROJECT_SPEC.md`
- **Current status:** `docs/RESEARCH_STATE.md`
- **Decisions:** `docs/DECISION_LOG.md`
- **Risks:** `docs/RISK_REGISTER.md`
- **Experiment records:** `PROJECT_CONTROL_CENTER.xlsx`

Do not silently change model architecture, split, metrics, or evaluation protocol in notebooks. Update the specification first and log the decision.

## Cloud-first execution
- GitHub = source code, notebooks, markdown, configs, lightweight result tables.
- Google Drive = dataset, model checkpoints, large plots, report, slides.
- Google Colab = primary runtime.
- Kaggle = optional compute backup, not a parallel project branch.
- Local machine = browser/editor only; no local GPU is required.

## Immediate first milestone
Complete **Gate 1 — Dataset Audit**. Do not train CNN models before the audit and split logic are approved.
