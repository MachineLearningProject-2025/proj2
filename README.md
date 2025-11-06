# [MLP Project 2] Predicting Human Preferences for LLM Response Enhancement

## 📌 1. Project Overview

| Detail                | Description                                                                                                                                                                                                           |
| :-------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Course**            | CS 53744 Machine Learning Project                                                                                                                                                                                     |
| **Task**              | Multiclass classification to predict **human preference** between two LLM responses (*A win*, *B win*, *Tie*).                                                                                                        |
| **Dataset**           | [Kaggle Competition – LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning)                                                                                               |
| **Goal**              | Build models that approximate **human judgment** in pairwise response evaluation through lexical, semantic, and contextual modeling.                                                                                  |
| **Evaluation Metric** | Log Loss (Kaggle official), Accuracy, Macro F1                                                                                                                                                                        |
| **Final Model**       | **DeBERTa-v3-small + LoRA Fine-Tuning + Isotonic Calibration**                                                                                                                                                        |
| **Baseline Models**   | Logistic Regression (Lexical only), SentenceTransformer all-MiniLM-L6-v2 (Semantic only), Hybrid Stacking (Lexical + Semantic)                                                                                        |
| **Key Insight**       | While hybrid ensembling achieved higher raw accuracy, the **DeBERTa + LoRA model** provided more balanced probability calibration and superior performance in **Tie prediction**, reflecting human-aligned reasoning. |


-----

## 👥 2. Team Information

| Role | Name | GitHub ID |
| :--- | :--- | :--- |
| Member | 박원규 | `@keiro23` |
| Member | 이유정 | `@yousrchive` |
| Member | 정승환 | `@whan0767` |

-----

## 🏆 3. Final Performance Summary

The final submitted model is based on **DeBERTa-v3-small** fine-tuned with **LoRA (Low-Rank Adaptation)** and post-processed through **Isotonic Regression calibration**.
This setup enables **token-level contextual comparison** between paired responses while preserving computational efficiency in the no-internet Kaggle GPU environment.

| Metric                     | Value                                              | Note                                           |
| :------------------------- | :------------------------------------------------- | :--------------------------------------------- |
| **Kaggle Public Log Loss** | **1.09**                                           | Final submission score on Kaggle leaderboard   |
| **Validation Accuracy**    | **0.42**                                           | Evaluated on 20% stratified validation split   |
| **Validation Macro F1**    | **0.42**                                           | Balanced across A-win, B-win, and Tie          |
| **Calibration Gain**       | **≈ +10% relative improvement in accuracy**        | Achieved through post-hoc isotonic calibration |
| **Final Model**            | **DeBERTa-v3-small + LoRA + Isotonic Calibration** | Fine-tuned checkpoint `checkpoint-22992`       |

**Key Strengths:**

* **Improved Tie recognition** through probabilistic calibration and contextual understanding
* **Lightweight training** using LoRA — only low-rank attention weights updated
* **Fully reproducible** in Kaggle’s no-internet environment (pre-mounted datasets and model weights)
* **Human-aligned reasoning:** prioritizes balanced, context-aware prediction rather than lexical bias

**Notes:**
Although the Hybrid Stacking model achieved slightly higher raw validation accuracy (0.48),
the DeBERTa + LoRA model exhibited superior alignment with human judgment,
especially in **contextually equivalent (Tie)** cases.
This makes it a more robust and scalable final model for preference modeling tasks.


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