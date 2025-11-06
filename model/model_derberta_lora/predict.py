# predict.py
# --------------------------------------------------
# Main script to generate the final submission.
# This script ties all utilities together.
# --------------------------------------------------

import os
import numpy as np
import torch
import pandas as pd
from config import *
from data_loader import load_data, create_text_features, get_validation_split
from model_loader import load_lora_model
from calibration import get_probabilities, train_calibrators, apply_calibration

def main():
    # 0. Disable W&B Logging
    os.environ["WANDB_DISABLED"] = "true"
    
    # 1. Load all data
    train_df, test_df, submission_df = load_data(DATA_PATH)
    
    # 2. Create 'text' features for both dataframes
    train_df = create_text_features(train_df)
    test_df = create_text_features(test_df)
    
    # 3. Get validation split (for calibration)
    val_texts, val_labels = get_validation_split(train_df)
    
    # 4. Load the fine-tuned LoRA model and tokenizer
    lora_model, tokenizer = load_lora_model(BASE_MODEL_NAME, LORA_CHECKPOINT_PATH)
    
    # 5. Train Calibrators
    # 5a. Get model probabilities for the validation set
    val_probs = get_probabilities(lora_model, tokenizer, val_texts, MAX_LEN)
    # 5b. Train IsotonicRegression models
    calibrators = train_calibrators(val_probs, val_labels)
    
    # 6. Get Predictions for Test Set
    # 6a. Get model probabilities for the test set
    test_probs = get_probabilities(lora_model, tokenizer, test_df['text'], MAX_LEN)
    # 6b. Apply trained calibrators
    normalized_probs = apply_calibration(test_probs, calibrators)
    
    # 7. Create Submission File
    print("➡️ Creating final submission file...")
    submission_df['winner_model_a'] = normalized_probs[:, 0]
    submission_df['winner_model_b'] = normalized_probs[:, 1]
    submission_df['winner_tie'] = normalized_probs[:, 2]

    submission_df.to_csv(OUTPUT_SUBMISSION_FILE, index=False)
    
    print(f"✅ Submission file saved to {OUTPUT_SUBMISSION_FILE}")
    print(submission_df.head())

if __name__ == "__main__":
    main()