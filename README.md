# Pattern Recognition Project – Part 2: Histopathology Cancer Detection

**Course:** Pattern Recognition (M.Sc.)  
**Instructor:** Prof. Raja Hashim Ali  
**Group:** WS25-PR19  
**Project:** Histopathology Cancer Detection (Microscope Images)

## Group Members & Roles

- **Student 1 (Technical Lead, Submitter):** Teyyub Malikov
  - Implements the complete technical pipeline
  - Conducts experiments and evaluations
  - Defines five (5) publication-ready research questions
  - Generates all quantitative results

- **Student 2 (Figures, Tables & Presentation):** Tahir Akhundov
  - Designs all figures (workflow, dataset overview, network architecture, result visualizations)
  - Prepares and maintains the presentation slides

- **Student 3 (Report & Storytelling):** Junior Ugochukwu Osuocha
  - Writes the complete report using the provided Overleaf template
  - Ensures a coherent narrative across sections
  - Aligns methodology, results, and discussion

## Project Overview

This project addresses automated cancer detection in histopathology images using a hybrid expert system combining:
- **Base Learner:** Convolutional Neural Networks (CNNs)
- **Meta-learner:** Ensemble methods for improved accuracy and calibration
- **Rule Engine:** Rule-based reasoning for interpretability and reliability

## Research Questions

The project addresses five (5) publication-ready research questions:

1. **RQ1:** How effective are Convolutional Neural Networks (CNNs) in detecting cancer in histopathology images compared to traditional machine learning approaches?

2. **RQ2:** What is the impact of metalearner integration on improving classification accuracy and confidence calibration in histopathology cancer detection?

3. **RQ3:** How can rule-based reasoning enhance the interpretability and reliability of automated cancer detection systems?

4. **RQ4:** What is the comparative performance of the hybrid expert system (CNN + Metalearner + Rule Engine) versus individual components?

5. **RQ5:** How generalizable are the developed models across different cancer types and histopathology image acquisition protocols?

## Dataset

**Note:** Large dataset files are not included in this repository due to size limitations. Please download them from the sources below and place them in the appropriate directories.

### Primary Dataset: PCam (PatchCamelyon)
- **Source:** [PCam Dataset](https://github.com/basveeling/pcam) and
[PCAM](https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon/data) 
 [YLung & Colon Cancer Histopathology Dataset)](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) - RQ5  

- **Description:** Histopathology patches from lymph node sections
- **Format:** H5 files containing 96x96 RGB patches
- **Download Instructions:**
  1. Download the following files from Kaggle or the official source:
     - `training_split.h5` → Place in `data/pcam/training_split.h5`
     - `validation_split.h5` → Place in `data/pcam/validation_split.h5`
     - `test_split.h5` → Place in `data/pcam/test_split.h5`
  2. Download label files:
     - `camelyonpatch_level_2_split_train_y.h5` → Place in `data/Labels/Labels/camelyonpatch_level_2_split_train_y.h5`
     - `camelyonpatch_level_2_split_valid_y.h5` → Place in `data/Labels/Labels/camelyonpatch_level_2_split_valid_y.h5`
     - `camelyonpatch_level_2_split_test_y.h5` → Place in `data/Labels/Labels/camelyonpatch_level_2_split_test_y.h5`

### External Datasets (for RQ5 - Generalization)

These datasets are used to evaluate model generalization across different cancer types and image acquisition protocols.

1. **Lung & Colon Cancer Histopathology Dataset**
   - **Download from:** [Kaggle - Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
   - **What to do:** Download the zip file, extract it, and place the extracted folder contents into `lung_colon_image_set/`
   - **Expected structure after extraction:**
     ```
     lung_colon_image_set/
     ├── colon_image_sets/
     │   ├── colon_aca/
     │   └── colon_n/
     └── lung_image_sets/
         ├── lung_aca/
         ├── lung_n/
         └── lung_scc/
     ```



## Model Architecture

### Base Learners

1. **Baseline CNN:** Custom convolutional neural network
   - Architecture:
     - Conv1: 3 → 16 channels, 3x3 kernel, padding=1
     - Conv2: 16 → 32 channels, 3x3 kernel, padding=1
     - Conv3: 32 → 64 channels, 3x3 kernel, padding=1
     - MaxPool2d (2x2) after each conv layer
     - FC1: 64×12×12 → 128
     - FC2: 128 → 2 (binary classification)
   - Activation: ReLU
   - Output: Binary classification (cancer/no cancer)

2. **ResNet18 (Transfer Learning):** Pre-trained ResNet18 with fine-tuning
   - Base: ImageNet pre-trained weights (ResNet18_Weights.DEFAULT)
   - Head: Custom classification head (fc layer: 512 → 2)
   - Training: Head-only or full fine-tuning
   - Framework: PyTorch torchvision

### Meta-learner
- **Ensemble Method:** Simple averaging of base learner probabilities
- **Calibration:** Temperature scaling for confidence calibration
- **Expected Calibration Error (ECE):** Used to evaluate calibration quality
- **Brier Score:** Used to evaluate probabilistic predictions

### Rule Engine
- **High Confidence Rule:** Accept predictions with very high/low ensemble probability (threshold: 0.90/0.10)
- **Uncertainty Handling:** Abstain when base models disagree
- **Normal Decision:** Follow ensemble decision for moderate confidence
- **Coverage:** Percentage of samples for which the rule engine makes a decision (vs. abstaining)

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Jupyter Notebook or JupyterLab

### Required Packages
```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install h5py
pip install openpyxl  # For Excel file handling
pip install Pillow
pip install tqdm
```

### Environment Setup

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd PR1.03
   ```

2. **Create the required folder structure:**
   ```bash
   mkdir -p data/pcam
   mkdir -p data/Labels/Labels
   mkdir -p lung_colon_image_set
   ```

3. **Download and place datasets:**
   
   **Step 3a: PCam Dataset (Primary Dataset)**
   - Go to: [Kaggle - PatchCamelyon Dataset](https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon/data)
   - Download these files:
     - `training_split.h5` → Save to `data/pcam/training_split.h5`
     - `validation_split.h5` → Save to `data/pcam/validation_split.h5`
     - `test_split.h5` → Save to `data/pcam/test_split.h5`
     - `camelyonpatch_level_2_split_train_y.h5` → Save to `data/Labels/Labels/camelyonpatch_level_2_split_train_y.h5`
     - `camelyonpatch_level_2_split_valid_y.h5` → Save to `data/Labels/Labels/camelyonpatch_level_2_split_valid_y.h5`
     - `camelyonpatch_level_2_split_test_y.h5` → Save to `data/Labels/Labels/camelyonpatch_level_2_split_test_y.h5`
   
   **Step 3b: Lung & Colon Cancer Dataset (for RQ5)**
   - Go to: [Kaggle - Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
   - Download the dataset zip file
   - Extract/unzip to `lung_colon_image_set/` folder (should contain `colon_image_sets/` and `lung_image_sets/` subfolders)
   

4. **Verify folder structure:**
   After downloading and placing all files, your project should have this structure:
   ```
   PR1.03/
   ├── data/
   │   ├── pcam/
   │   │   ├── training_split.h5
   │   │   ├── validation_split.h5
   │   │   └── test_split.h5
   │   ├── Labels/
   │   │   └── Labels/
   │   │       ├── camelyonpatch_level_2_split_train_y.h5
   │   │       ├── camelyonpatch_level_2_split_valid_y.h5
   │   │       └── camelyonpatch_level_2_split_test_y.h5
   │   └── external/
   │       ├── BreakHis/
   │       └── LC25000/
   └── lung_colon_image_set/
       ├── colon_image_sets/
       │   ├── colon_aca/
       │   └── colon_n/
       └── lung_image_sets/
           ├── lung_aca/
           ├── lung_n/
           └── lung_scc/
   ```

5. **Open the notebook:**
   - Open `Part2_Histopathology_Cancer_Detection.ipynb` in Jupyter Notebook or JupyterLab

6. **Run the notebook:**
   - Execute all cells sequentially (the notebook is designed to run end-to-end)

## Usage

### Running the Notebook
1. Open `Part2_Histopathology_Cancer_Detection.ipynb`
2. Execute cells in order (the notebook is designed to run end-to-end)
3. Results will be automatically saved to `Figures_Tables/` directory

### Expected Outputs

The notebook generates:
- **Model checkpoints:** Saved in `outputs/` directory
  - `baseline_cnn_full.pth` - Trained baseline CNN
  - `resnet18_tl_headonly.pth` - Fine-tuned ResNet18
- **Figures:** PDF format in `Figures_Tables/RQx/` (one folder per research question)
- **Tables:** Excel format in `Figures_Tables/RQx/` (one folder per research question)

### File Naming Convention
- **Figures:** `RQx_FigN.pdf` (e.g., `RQ1_Fig1.pdf`, `RQ2_Fig1.pdf`)
- **Tables:** `RQx_TabN.xlsx` (e.g., `RQ1_Tab1.xlsx`, `RQ2_Tab1.xlsx`)

## Project Structure

```
PR1.03/
├── Part2_Histopathology_Cancer_Detection.ipynb  # Main notebook
├── README.md                                      # This file
├── .gitignore                                     # Git ignore rules
├── data/
│   ├── pcam/                                     # PCam dataset (H5 image files)
│   │   ├── training_split.h5
│   │   ├── validation_split.h5
│   │   └── test_split.h5
│   ├── Labels/                                   # PCam label files (H5)
│   │   └── Labels/
│   │       ├── camelyonpatch_level_2_split_train_y.h5
│   │       ├── camelyonpatch_level_2_split_valid_y.h5
│   │       └── camelyonpatch_level_2_split_test_y.h5
│   ├── Metadata/                                 # Metadata CSV files (optional)
│   └── train/                                    # Organized training images (optional)
├── Figures_Tables/                               # All outputs organized by RQ
│   ├── RQ1/                                     # RQ1 figures and tables
│   ├── RQ2/                                     # RQ2 figures and tables
│   ├── RQ3/                                     # RQ3 figures and tables
│   ├── RQ4/                                     # RQ4 figures and tables
│   └── RQ5/                                     # RQ5 figures and tables
├── outputs/                                      # Model checkpoints (not in repo)
└── lung_colon_image_set/                         # External dataset for RQ5
    ├── colon_image_sets/
    └── lung_image_sets/
```

## Reproducing Results

1. **Data Preparation:**
   - Ensure all datasets are in the correct directories
   - PCam dataset should be in H5 format with images in `data/pcam/` and labels in `data/Labels/Labels/`

2. **Model Training:**
   - Run the notebook cells sequentially
   - Training will automatically save checkpoints to `outputs/`
   - If checkpoints exist, the notebook will load them instead of retraining

3. **Evaluation:**
   - Each research question section evaluates models and saves results
   - Results are automatically saved with proper naming convention

4. **Output Generation:**
   - All figures are saved as PDF
   - All tables are saved as Excel (.xlsx)
   - Files are automatically organized in RQ-specific folders

## Key Features

- **End-to-end Pipeline:** Complete workflow from data loading to evaluation
- **Modular Design:** Separate sections for each research question
- **Automatic Organization:** Figures and tables automatically saved with proper naming
- **Reproducibility:** Checkpoint saving/loading for consistent results
- **Comprehensive Evaluation:** Multiple metrics (accuracy, precision, recall, F1, calibration, coverage)

## Experimental Results Summary

- **RQ1:** Baseline CNN performance on histopathology cancer detection
- **RQ2:** Ensemble (meta-learner) performance with calibration analysis
- **RQ3:** Rule-based reasoning for interpretability and reliability
- **RQ4:** Hybrid expert system (CNN + Ensemble + Rules) performance
- **RQ5:** Cross-protocol generalization across different cancer types

## Notes
--- **Kernel Selection:** When opening the notebook, if you see a kernel selection prompt, choose any available **Python 3** kernel. The notebook was created with a custom kernel name (`histo311`) but will work with any standard Python 3 environment that has the required packages installed.
- The notebook is designed to run end-to-end without errors
- Model checkpoints are saved to avoid retraining
- All outputs follow the submission requirements (PDF for figures, Excel for tables)
- The notebook handles missing variables gracefully with try-except blocks


This repository contains:
-  Complete Jupyter Notebook (`.ipynb`)
-  Well-documented code with group information and research questions
-  Dataset loading and preprocessing
-  Model implementation (base learner, meta-learner, rule engine)
-  Experiments answering all five research questions
-  Properly organized figures and tables

## License

This project is part of an academic course assignment.

## Contact

For questions or issues, please contact the Technical Lead (Student 1): Teyyub Malikov

---

**Last Updated:** January 2026



