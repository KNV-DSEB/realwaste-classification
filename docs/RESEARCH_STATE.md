# RESEARCH_STATE

**Status date:** Day 0 / Starter Pack generated

## Problem
Multi-class real-world waste image classification.

## Dataset
RealWaste selected from Kaggle. Dataset audit pending.

## Question
Compare simple custom CNN, complex custom CNN, and transfer learning under a fixed protocol.

## Locked design
- E1 SimpleCNN
- E2 MultiScaleCNN
- E3 EfficientNet-B0
- E4 optional augmentation ablation
- 224×224
- stratified 70/15/15 split
- seed 42

## Completed
- scope selected
- architecture families selected
- 10-day plan defined
- starter pack initialized

## Current gate
**Gate 0 — Setup**

## Next gate
**Gate 1 — Dataset Audit**

## Open unknowns
- actual number of usable images in downloaded Kaggle copy;
- actual class folders/class mapping;
- corrupted/unreadable images;
- dimension/mode distribution;
- duplicate/near-duplicate risk;
- observed imbalance severity;
- exact cloud paths for the team.

## Rule
Do not update this file with expected results. Only record verified project state.
