# --------------------------------------------------
# Functions for loading and preparing data.
# --------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import VAL_TEST_SIZE, RANDOM_SEED

def load_data(data_path):
    """
    Loads train, test, and sample submission CSVs.
    """
    print("➡️ Loading data...")
    try:
        train_df = pd.read_csv(data_path + 'train.csv')
        test_df = pd.read_csv(data_path + 'test.csv')
        submission_df = pd.read_csv(data_path + 'sample_submission.csv')
        print("CSV files loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: Files not found at {data_path}. Trying local path '../dataset/'...")
        data_path = '../dataset/' # Fallback path
        train_df = pd.read_csv(data_path + 'train.csv')
        test_df = pd.read_csv(data_path + 'test.csv')
        submission_df = pd.read_csv(data_path + 'sample_submission.csv')
        print("Local CSV files loaded.")
    
    return train_df, test_df, submission_df

def create_text_features(df):
    """
    Creates the combined 'text' column using the P+A+B simple concatenation.
    """
    print("Creating 'text' feature column...")
    def create_text(row):
        return (
            f"prompt: {str(row['prompt'])}"
            f"\n\nresponse_a: {str(row['response_a'])}"
            f"\n\nresponse_b: {str(row['response_b'])}"
        )
    df['text'] = df.apply(create_text, axis=1)
    return df

def get_validation_split(train_df):
    """
    Applies labeling and splitting to the train_df to get a validation set
    for calibration.
    """
    print("Creating validation split for calibration...")
    # 1. Create labels
    conditions = [
        train_df['winner_model_a'] == 1, 
        train_df['winner_model_b'] == 1, 
        train_df['winner_tie'] == 1
    ]
    choices = [0, 1, 2] # (A_win, B_win, Tie)
    train_df['label'] = np.select(conditions, choices, default=-1)

    # 2. Filter invalid labels
    train_df = train_df[train_df['label'] != -1].copy()

    # 3. Create the *exact* validation split used in training
    _, val_texts, _, val_labels = train_test_split(
        train_df['text'], 
        train_df['label'], 
        test_size=VAL_TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_df['label']
    )
    print(f"Validation set created with {len(val_texts)} samples.")
    return val_texts, val_labels