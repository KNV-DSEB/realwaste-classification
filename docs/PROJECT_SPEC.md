# PROJECT_SPEC — LOCKED v1.1

v1.1 (2026-10-02): §6, §7, §8 (E2), §9 and §11 filled in from DECISION_LOG D007–D013. No change to models, split ratios, or metrics.
v1.2 (2026-10-03): §6 split grouped by consecutive file numbers (D014, D015). Ratios, seed, models and metrics unchanged.

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
- Split: stratified 70% train / 15% validation / 15% test, grouped by blocks of 10 consecutive file numbers per class so that re-shots of the same item stay in one split (D015, replaces the image-level split D007 after the duplicate check D014).
- Random seed: 42.
- Split output: `split_grouped_v2.csv` containing filepath, class label, block, and split assignment.
- One fixed split reused by every core experiment, pinned by SHA256 in `configs/config.yaml`; load it only through `src/dataset.py`.
- Corrupt images are removed before split and documented.
- Near-duplicate risk is inspected during audit; if material duplicates are found, resolve before freezing split. Check: `notebooks/03_duplicate_check.ipynb`.

## 7. Preprocessing
Validation/Test:
- RGB conversion
- deterministic resize to 224 × 224 (images are 524 × 524 squares, so no crop is needed)
- normalization with ImageNet mean/std for every model (D008)

Training augmentation (D011; identical for E1–E3, disabled for E4):
- RandomResizedCrop to 224, scale 0.8–1.0
- Random horizontal flip (p = 0.5)
- Small random rotation (±10°)
- Mild brightness/contrast/saturation jitter (0.2 each, no hue shift)

No random augmentation on validation/test. Exact values live in `configs/config.yaml`; the implementation is `src/transforms.py`.

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
Custom complex CNN from scratch. Each multi-scale block contains parallel convolution branches with different receptive fields (e.g. 1×1, 3×3, 5×5), concatenated then normalized/activated. Use multiple blocks with max-pooling downsampling between blocks, followed by global average pooling and a fully connected head (Dense → ReLU → Dropout → Dense(K)), so the model explicitly contains convolution, pooling and fully connected components (D013).

Purpose: test whether multi-scale feature extraction improves real-world waste classification.

### E3 — EfficientNet-B0
ImageNet-pretrained EfficientNet-B0:
- Stage A: freeze backbone and train new classification head.
- Stage B: unfreeze selected late blocks and fine-tune with a lower learning rate.

Purpose: assignment-required transfer learning/fine-tuning model.

### E4 — Optional augmentation ablation
Repeat E2 without training augmentation, with all other settings held as close as possible.

## 9. Loss / imbalance handling
Default loss: Cross-Entropy, unweighted, for all core models (D009).

The audit measured a moderate 2.90× imbalance (921 Plastic vs 318 Textile Trash). Class weighting is not enabled by default. It is introduced only if validation results show minority-class recall collapsing; if so, the same policy is applied to all core models and logged.

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
Use validation performance for model/checkpoint selection. Checkpoint selection and early stopping both use validation Macro F1 (mode max, patience 5) for all core models (D010). E3: Stage B starts from the best Stage-A checkpoint, and the final E3 checkpoint is the best validation Macro F1 over both stages. The test set is evaluated once per model, after selection.

## 12. Result table contract
Every core experiment must produce:
`experiment_id, model_name, seed, best_epoch, accuracy, macro_precision, macro_recall, macro_f1, parameter_count, training_time_minutes, checkpoint_path, notes`

## 13. Stop conditions
- If a core model fails to run by Day 6, reduce implementation complexity before adding new experiments.
- If time is short, drop E4, Grad-CAM, demo, and extra models before compromising E1–E3 or report quality.
