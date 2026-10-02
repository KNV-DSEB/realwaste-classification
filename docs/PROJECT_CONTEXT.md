# PROJECT_CONTEXT

## Course context
This is a Deep Learning major assignment completed by a four-person team under a 10-day deadline.

The lecturer requires the project to use an image dataset for a classification or segmentation task and to include:
1. Data preprocessing, cleaning, normalization, augmentation, and Train/Validation/Test split.
2. A simple CNN designed by the team using convolution, pooling, and fully connected layers.
3. A more complex CNN designed by the team using CNN blocks and related layers.
4. A transfer-learning/fine-tuning CNN model.
5. Test-set evaluation using appropriate metrics.
6. A report containing CNN theory, problem description, dataset description, model architecture, experiments, conclusion, and references.

## Team decision
Dataset: RealWaste from Kaggle (user-selected dataset link):
https://www.kaggle.com/datasets/joebeachcapital/realwaste/data

The exact class counts, file counts, dimensions, and corruption status MUST be measured from the downloaded copy before being written into the report.

## Why this scope
The project prioritizes execution quality under a 10-day deadline. The dataset is small enough for cloud training while still representing visually challenging real-world waste conditions. The design creates a clean progression from simple CNN to custom complex CNN to transfer learning.

## Runtime strategy
Primary: Google Colab
Storage: Shared Google Drive
Version control: GitHub
Backup compute: Kaggle Notebook if needed

## Deliverables
- Working reproducible codebase
- Fixed data split
- Three core trained/evaluated models
- Result tables and plots
- Error analysis
- Major Assignment report
- Presentation slides
- Optional demo only after core deliverables are safe
