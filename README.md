# 🤖 MLP Team Project 2: Predicting Human Preferences for LLM Response Enhancement

## 📌 1. Project Overview

| Detail | Description |
| :--- | :--- |
| **Course** | CS 53744 Machine Learning Project |
| **Task** | Multiclass classification to predict human preference (A win, B win, Tie) for LLM responses. |
| **Dataset** | Kaggle Competition - LLM Classification Finetuning |
| **Evaluation Metric** | Log Loss |
| **Due Date** | 11:59 PM, November 6, 2025 |
| **Final Model** | **Hybrid Stacking Classifier** (MiniLM Embeddings + Scaled Lexical Features) |

-----

## 👥 2. Team Information

| Role | Name | GitHub ID |
| :--- | :--- | :--- |
| Member | \[박원규] | `[keiro23]` |
| Member | \[이유정] | `[yousrchive]` |
| Member | \[정승환] | `[whan0767]` |

-----

## 🏆 3. Final Performance Summary

The final model utilizes a combination of difference-based **lexical features** and **Sentence Transformer embeddings** to achieve a robust prediction.

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Kaggle Public Score** | $\text{[Final Log Loss Value]}$ | Screenshot provided in the PDF Report. |
| **Validation Log Loss** | $\text{[Validation Log Loss Value]}$ | Achieved via Grid Search on 20% validation split. |
| **Key Techniques** | MiniLM-L6-v2 Embeddings, Feature Scaling (Robust/Standard), Stacking Ensemble (LR, RF, LGBM). |

-----

## ⚙️ 4. How to Reproduce Results

This guide provides the steps to reproduce the final model training and generate the submission file using the provided Python modules.

### 4.1. Environment Setup & Dependencies

1.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    ```
2.  **Install Required Packages:** Install all dependencies listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### 4.2. Data Preparation

1.  **Download Files:** Obtain the three CSV files from the Kaggle competition page.
2.  **Place Data:** Save the downloaded files (`train.csv`, `test.csv`, `sample_submission.csv`) into the **`dataset/`** folder within the `PROJ2` root directory.

### 4.3. Execution Pipeline

The entire workflow is managed by `main.py`, which orchestrates feature engineering, embedding generation, model training, and prediction.

1.  **Change Directory:** Navigate to the folder containing the main execution script (`main.py`) to ensure relative paths are correctly resolved.
    ```bash
    cd notebook
    ```
2.  **Create Submission Folder:** (If it doesn't exist yet)
    ```bash
    mkdir ../submission
    ```
3.  **Run Main Script:** Execute the full pipeline. **Note:** This step involves downloading the MiniLM model and running Grid Search, which may take significant time.
    ```bash
    python main.py
    ```

### 4.4. Pipeline Workflow

The code is modularized for clarity and reproducibility:

  * **`data_loader.py`**: Handles CSV loading and initial target variable creation.
  * **`feature_engineering.py`**: Calculates difference-based features (e.g., `len_diff`, `punc_diff`, `lexical_div_diff`, keyword presence) and applies a **ColumnTransformer** for feature scaling.
  * **`model_trainer.py`**: Generates **`all-MiniLM-L6-v2`** embeddings for all Prompt+Response pairs, combines them with the scaled tabular features, and trains the **Stacking Classifier** (with internal $\text{3-Fold}$ cross-validation and hyperparameter search).
  * **Output**: The script saves the final predictions to **`../submission/final_submission.csv`**.

-----

## 📁 5. Project Directory Structure

```
PROJ2/
├── dataset/                     # Input Data Location
│   ├── train.csv
│   └── test.csv
├── submission/                  # Output Submission File Location
│   └── final_submission.csv
├── notebook/                    # Main Code Modules
│   ├── **data_loader.py**
│   ├── **feature_engineering.py**
│   ├── **model_trainer.py**
│   └── **main.py** # 👈 Start the execution here
├── experiments/                 # Jupyter Notebooks for Intermediate Steps
├── .venv/                       # Python Virtual Environment
└── **requirements.txt** # All Python Dependencies
```