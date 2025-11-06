# --------------------------------------------------
# Function for loading the fine-tuned PEFT model.
# --------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from config import NUM_LABELS

def load_lora_model(base_model_path, lora_checkpoint_path):
    """
    Loads the base model, applies the LoRA checkpoint, and returns 
    the merged model and its tokenizer.
    """
    print("➡️ Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path, 
        num_labels=NUM_LABELS, 
        device_map='auto'
    )

    print(f"Loading LoRA adapter from: {lora_checkpoint_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    lora_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)
    print(f"Model is ready on device: {device}")
    
    return lora_model, tokenizer