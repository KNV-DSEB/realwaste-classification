# TEAM_OPERATING_SYSTEM

## Roles
### Data Lead
Owns dataset audit, class mapping, split.csv, preprocessing verification, and dataset documentation.

### Simple CNN Lead
Owns E1 implementation, training record, architecture diagram, and relevant CNN theory notes.

### Complex CNN Lead
Owns E2 MultiScaleCNN implementation, architecture rationale, and E4 if time permits.

### Transfer/Evaluation Lead
Owns E3 EfficientNet-B0, shared evaluation pipeline, final confusion matrices, and consolidated result table.

### Integration Lead
One team member must also own integration. Responsibilities:
- stable repo structure;
- shared config;
- same split;
- consistent result filenames;
- merge conflict resolution;
- release tagging before submission.

## Git discipline
Suggested branches:
- `main` — stable only
- `data-audit`
- `model-simple`
- `model-multiscale`
- `model-efficientnet`
- `report`

No direct large binary uploads to GitHub.

## Drive discipline
Suggested shared Drive:
`Deep Learning - RealWaste/`
- `01_Dataset/`
- `02_Splits/`
- `03_Checkpoints/`
- `04_Results/`
- `05_Report/`
- `06_Presentation/`

Each experiment has its own checkpoint/result folder.

## Naming convention
Defined in `RESULT_CONTRACT.md` (D012).

Checkpoints:
- `03_Checkpoints/E1/E1_best.pt`
- `03_Checkpoints/E2/E2_best.pt`
- `03_Checkpoints/E3/E3_best.pt`

Result files (in `04_Results/experiments/E1/`):
- `E1_metrics.json`
- `E1_history.csv`
- `E1_confusion_matrix.png`

Avoid `final2`, `final_final`, `best_real` filenames.

## Daily stand-up — 10 minutes
Each member answers:
1. What did I complete?
2. What evidence/output exists?
3. What is blocked?
4. What will I finish today?
5. Did I change anything affecting PROJECT_SPEC?
