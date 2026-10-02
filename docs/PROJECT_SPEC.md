# PROJECT_SPEC — LOCKED v1.0

## 1. Title
**Robust Real-World Waste Classification using Custom CNN Architectures and Transfer Learning**

## 2. Problem type
Multi-class image classification.

## 3. Input / Output
**Input:** RGB waste image.

**Output:** Predicted waste class and class probabilities.

The class list and number of classes are read from the audited dataset. Do not hard-code a class list before Gate 1.

## 4. Research objective
Evaluate how CNN architectural complexity and transfer learning affect waste-classification performance under one controlled experimental protocol.

## 5. Research questions
**RQ1.** How well does a simple CNN trained from scratch classify RealWaste images?

**RQ2.** Does a custom complex Multi-Scale CNN improve performance over the simple CNN under the same data protocol?

**RQ3.** How much does ImageNet transfer learning with EfficientNet-B0 improve performance compared with CNNs trained from scratch?

**RQ4 (optional).** Does training-time data augmentation improve generalization for the custom Multi-Scale CNN?

## 6. Data protocol
- Source: RealWaste Kaggle dataset selected by the team.
- Image size: 224 × 224 RGB.
- Split: stratified 70% train / 15% validation / 15% test.
- Random seed: 42.
- Split output: `split.csv` containing filepath, class label, and split assignment.
- One fixed split reused by every core experiment.
- Corrupt images are removed before split and documented.
- Near-duplicate risk is inspected during audit; if material duplicates are found, resolve before freezing split.

## 7. Preprocessing
Validation/Test:
- RGB conversion
- deterministic resize/crop to 224 × 224
- normalization compatible with the model setup

Training augmentation (initial plan):
- Random horizontal flip
- Small random rotation
- Random resized crop or resize + crop
- Mild brightness/contrast/color jitter

No random augmentation on validation/test.

## 8. Core models
### E1 — SimpleCNN
Custom model from scratch:
- Conv(32) → BN → ReLU → MaxPool
- Conv(64) → BN → ReLU → MaxPool
- Conv(128) → BN → ReLU → MaxPool
- Global Average Pooling
- Dense(128) → ReLU → Dropout
- Dense(K) → Softmax/logits

Purpose: simple baseline required by assignment.

### E2 — MultiScaleCNN
Custom complex CNN from scratch. Each multi-scale block contains parallel convolution branches with different receptive fields (e.g. 1×1, 3×3, 5×5), concatenated then normalized/activated. Use multiple blocks followed by global pooling and a classification head.

Purpose: test whether multi-scale feature extraction improves real-world waste classification.

### E3 — EfficientNet-B0
ImageNet-pretrained EfficientNet-B0:
- Stage A: freeze backbone and train new classification head.
- Stage B: unfreeze selected late blocks and fine-tune with a lower learning rate.

Purpose: assignment-required transfer learning/fine-tuning model.

### E4 — Optional augmentation ablation
Repeat E2 without training augmentation, with all other settings held as close as possible.

## 9. Loss / imbalance handling
Default loss: Cross-Entropy.

Class weighting is NOT automatically enabled. Decide after Dataset Audit. If imbalance is material, document the rationale and apply the same policy fairly where appropriate.

## 10. Metrics
Required main metrics:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1

Secondary diagnostics:
- Per-class Precision/Recall/F1
- Confusion matrix
- Train/validation loss curves
- Train/validation accuracy curves
- Parameter count
- Best epoch
- Training time, if reliably measured

## 11. Model selection
Use validation performance for model/checkpoint selection. Prefer Macro F1 as the primary selection metric if class imbalance is meaningful; otherwise Accuracy + Macro F1 are reported jointly.

## 12. Result table contract
Every core experiment must produce:
`experiment_id, model_name, seed, best_epoch, accuracy, macro_precision, macro_recall, macro_f1, parameter_count, training_time_minutes, checkpoint_path, notes`

## 13. Stop conditions
- If a core model fails to run by Day 6, reduce implementation complexity before adding new experiments.
- If time is short, drop E4, Grad-CAM, demo, and extra models before compromising E1–E3 or report quality.
