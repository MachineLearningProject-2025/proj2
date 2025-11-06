# calibration.py
# --------------------------------------------------
# Functions for training and applying calibration.
# --------------------------------------------------

import torch
import numpy as np
from config import NUM_LABELS
from sklearn.isotonic import IsotonicRegression
from transformers import Trainer, TrainingArguments

# Helper class for predictions
class TempDataset(torch.utils.data.Dataset):
    def __init__(self, encodings): 
        self.encodings = encodings
    def __getitem__(self, idx): 
        return {key: val[idx] for key, val in self.encodings.items()}
    def __len__(self): 
        return len(self.encodings['input_ids'])

def get_probabilities(model, tokenizer, texts, max_len):
    """
    Tokenizes texts and runs prediction to get softmax probabilities.
    """
    print(f"Tokenizing {len(texts)} texts...")
    encodings = tokenizer(
        texts.tolist(), 
        truncation=True, 
        padding=True, 
        max_length=max_len, 
        return_tensors="pt"
    )
    dataset = TempDataset(encodings)
    
    # We need a dummy trainer to run predictions
    # This disables wandb logging
    dummy_args = TrainingArguments(output_dir="./dummy_results", report_to="none")
    dummy_trainer = Trainer(model=model, args=dummy_args)
    
    print("Running model predictions...")
    predictions = dummy_trainer.predict(dataset)
    
    # Apply softmax to logits
    probs = torch.nn.functional.softmax(
        torch.from_numpy(predictions.predictions), 
        dim=-1
    ).numpy()
    
    return probs

def train_calibrators(val_probs, val_labels):
    """
    Trains one IsotonicRegression model for each class.
    """
    print("Training calibration models...")
    calibrators = {}
    for i in range(NUM_LABELS): # 0, 1, 2
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        # Create binary target for this class
        y_cal = (val_labels.to_numpy() == i).astype(int)
        
        # Fit calibrator
        iso_reg.fit(val_probs[:, i], y_cal)
        calibrators[i] = iso_reg
    
    print("Calibration models trained.")
    return calibrators

def apply_calibration(test_probs, calibrators):
    """
    Applies trained calibrators to test probabilities and normalizes them.
    """
    print("Applying calibration to test predictions...")
    calibrated_probs = np.zeros_like(test_probs)
    for i in range(NUM_LABELS):
        calibrated_probs[:, i] = calibrators[i].predict(test_probs[:, i])

    # Normalize probabilities to sum to 1
    calibrated_probs_sum = np.sum(calibrated_probs, axis=1, keepdims=True)
    normalized_probs = calibrated_probs / (calibrated_probs_sum + 1e-9)
    
    return normalized_probs