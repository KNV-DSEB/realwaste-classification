# EXPERIMENT_PROTOCOL

## Core experiment IDs
### E1 — SimpleCNN
Question: How far can a basic custom CNN go?

### E2 — MultiScaleCNN
Question: Does custom multi-scale feature extraction improve classification?

### E3 — EfficientNet-B0
Question: How much does pretrained representation + fine-tuning improve performance?

### E4 — MultiScaleCNN_NoAug (optional)
Question: What is the contribution of training augmentation?

## Fair comparison contract
For E1–E3:
- same split.csv;
- same class mapping;
- same input resolution;
- same final metric implementation;
- test evaluated only after validation-based selection;
- same result schema.

## Training records
Record for every run:
- experiment ID / run ID;
- git commit hash if available;
- seed;
- batch size;
- optimizer;
- initial LR;
- scheduler;
- epochs planned;
- epochs completed;
- early-stopping rule;
- augmentation profile;
- class-weight policy;
- best validation metric;
- checkpoint path;
- runtime/hardware if available.

## Primary reporting
Main comparison table:
Model | Accuracy | Macro Precision | Macro Recall | Macro F1

## Interpretation rules
Do not write “better” based on one metric without naming the metric and size of difference. Do not claim architecture caused a gain if multiple settings changed simultaneously.
