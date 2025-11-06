# train.py
# --------------------------------------------------
# STAGE 1: 데이터 전처리, 모델 훈련, 최종 모델 저장
# --------------------------------------------------

import os
import datasets
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig

# 설정 및 유틸리티 함수 임포트
from config import *
from utils import get_strategic_truncate_processor, get_dataset_splits

def main_train():
    print("🚀 STAGE 1: Starting Model Training...")
    
    # 0. (필수) 병렬 처리 비활성화
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.disable_multiprocessing()

    # 1. 데이터 로드
    try:
        raw_dataset = load_dataset("csv", data_files=DATA_PATH + "train.csv")
        print("Dataset loaded.")
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return

    # 2. 토크나이저 로드
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
    except Exception as e:
        print(f"Tokenizer 로드 실패: {e}")
        return
    print("Tokenizer loaded.")
    
    # 3. 데이터 전처리 및 분할
    processor = get_strategic_truncate_processor(tokenizer)
    train_dataset, val_dataset = get_dataset_splits(raw_dataset, processor)

    # 4. 모델 및 LoRA 설정
    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=NUM_LABELS, device_map="auto"
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="SEQ_CLS",
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    # 5. 훈련 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        fp16=True, # (FP16/BF16 자동 감지 대신 True로 고정, 필요시 수정)
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGGING_DIR,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 6. 훈련 시작
    print("Starting training...")
    trainer.train()
    print("Training complete.")
    
    # 7. 훈련된 모델 저장
    print(f"Saving trained model and tokenizer to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR) # LoRA 어댑터와 설정 저장
    tokenizer.save_pretrained(OUTPUT_DIR) # 토크나이저 파일 저장
    print("✅ STAGE 1: Training and saving complete.")


if __name__ == "__main__":
    main_train()