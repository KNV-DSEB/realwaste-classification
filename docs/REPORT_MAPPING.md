# REPORT_MAPPING — Lecturer Requirements → Project Evidence

## Part 1: Theory
Cover:
- CNN fundamentals
- convolution/filter/kernel
- feature maps
- activation functions
- pooling
- fully connected layers / global pooling
- loss and optimization
- regularization
- data augmentation
- transfer learning / fine-tuning

Owner: Simple CNN Lead + review by team.

## Part 2 — I. Problem Description
Include:
- title
- research objectives
- input
- output
- summary of tasks performed

Evidence source: PROJECT_SPEC + completed workflow.

## II. Dataset Description
Include:
- dataset link
- actual audited number of images
- class names and counts
- actual image sizes/modes
- sample images
- class distribution

Evidence source: Dataset Audit only. Never copy unverified online counts.

## III. CNN Model Design
### 1. Preprocess
- cleaning
- normalization
- augmentation
- exact Train/Val/Test counts

### 2. Simple CNN
- architecture diagram
- layer explanation
- parameter count

### 3. Complex CNN
- MultiScaleCNN diagram
- multi-scale rationale
- blocks/layers

### 4. Transfer Learning
- EfficientNet-B0 architecture role
- frozen-head phase
- fine-tuning phase

## IV. Experimental Results
Minimum:
- test metrics table
- Accuracy / Precision / Recall / F1
Recommended:
- Macro averaging explanation
- confusion matrices
- per-class metrics
- training curves
- E4 ablation if completed
- error analysis

## V. Conclusion
Report only supported findings:
- strongest observed model result;
- what comparison shows;
- major failure modes;
- limitations;
- practical implication without overclaiming.

## References
Verify every citation. Never cite a source not actually inspected.
