# CLOUD_WORKSPACE_SETUP

## Recommended architecture
**GitHub** = source code / notebooks / config / markdown / lightweight results.  
**Google Drive** = dataset / frozen split / checkpoints / large results / report / slides.  
**Google Colab** = primary runtime.  
**Kaggle Notebook** = compute backup only, using the same GitHub commit, config, and split.csv.

## Shared Drive folder
Create one team folder:

`Deep Learning - RealWaste/`

Subfolders:
- `01_Dataset/RealWaste/`
- `02_Splits/`
- `03_Checkpoints/E1/`, `03_Checkpoints/E2/`, `03_Checkpoints/E3/` (`E4/` if run) — naming per D012
- `04_Results/dataset_audit/`
- `04_Results/experiments/E1/` … `E4/`
- `05_Report/`
- `06_Presentation/`

## GitHub repository
Recommended private repo name:
`realwaste-classification`

Copy this starter pack into the repository root. Keep dataset/checkpoints out of Git using `.gitignore`.

## Colab rule
Every notebook should:
1. mount Drive;
2. print runtime/GPU;
3. set one `DRIVE_ROOT` variable;
4. import shared project code where applicable;
5. save checkpoints/results to Drive, not only `/content`;
6. record experiment ID and config.

## Security
Do not commit or paste GitHub/Kaggle access tokens into shared notebooks. Use secure authentication methods supported by the platform.

## Environment consistency
When using Kaggle as backup compute, do not create a separate pipeline. Use the same repository version, frozen split, image size, seed, metrics, and config.
