# 🤖 MLP Team Project 2: Predicting Human Preferences for LLM Response Enhancement

## 📌 1. Project Overview

| Detail | Description |
| :--- | :--- |
| **Course** | CS 53744 Machine Learning Project |
| **Task** | Multiclass classification to predict human preference (A win, B win, Tie) for LLM responses. |
| **Dataset** | Kaggle Competition - LLM Classification Finetuning |
| **Evaluation Metric** | Log Loss, accuracy score |
| **Final Model** | **Hybrid Stacking Classifier** (MiniLM Embeddings + Scaled Lexical Features) |

-----

## 👥 2. Team Information

| Role | Name | GitHub ID |
| :--- | :--- | :--- |
| Member | \박원규 | `@keiro23` |
| Member | \이유정 | `@yousrchive` |
| Member | \정승환 | `@whan0767` |

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

This guide provides the steps to reproduce the results for the two final models.

### 4.1. Environment Setup & Dependencies

1.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    ```
2.  **Install Required Packages:** Install all dependencies from the centralized `requirements.txt` file in the project root.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download NLTK Data:** Some lexical features depend on NLTK. Run this command in a Python interpreter to download the necessary data:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    ```

### 4.2. Data Preparation

1.  **Download Files:** Obtain the three CSV files from the Kaggle competition page.
2.  **Place Data:** Save the downloaded files (`train.csv`, `test.csv`, `sample_submission.csv`) into the **`dataset/`** folder within the `PROJ2` root directory.

### 4.3. Model 1: Hybrid Stacking Classifier

This model combines lexical features and sentence embeddings. The entire workflow is managed by `main.py`.

1.  **Change Directory:** Navigate to the script's location.
    ```bash
    cd model/model_hybrid_stacking
    ```
2.  **Run Main Script:** Execute the full pipeline. This will generate a submission file in the same directory.
    ```bash
    python main.py
    ```
3.  **Output:** The script saves predictions to `submission.csv`.

### 4.4. Model 2: DeBERTa with LoRA Fine-Tuning

This model fine-tunes a `DeBERTa-v3-small` model using LoRA. The process involves two stages: training and prediction.

1.  **Change Directory:** Navigate to the scripts' location.
    ```bash
    cd model/model_derberta_lora
    ```
2.  **Stage 1: Train the Model:**
    Execute the training script. This will fine-tune the model and save the LoRA adapter in the `results_lora_strategic/` directory.
    ```bash
    python train.py
    ```
3.  **Stage 2: Generate Predictions:**
    After training is complete, run the prediction script. This uses the trained LoRA weights and generates a submission file.
    ```bash
    python predict.py
    ```
4.  **Output:** The script saves predictions to `submission.csv`.

-----

## 📁 5. Project Directory Structure

```
PROJ2/
├── dataset/                     # Input Data Location
│   ├── train.csv
│   └── test.csv
├── model/
│   ├── model_derberta_lora/     # DeBERTa LoRA Model
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── config.py
│   │   └── ...
│   └── model_hybrid_stacking/   # Hybrid Stacking Model
│       ├── main.py
│       ├── model_trainer.py
│       └── ...
├── experiments/                 # Jupyter Notebooks for Intermediate Steps
├── .venv/                       # Python Virtual Environment
└── requirements.txt             # All Python Dependencies
```