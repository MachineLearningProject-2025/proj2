# utils.py
# --------------------------------------------------
# 훈련과 분석에 공통으로 사용되는 전처리 함수를 정의합니다.
# --------------------------------------------------

import numpy as np
from datasets import ClassLabel
from config import (
    MAX_LEN, PROMPT_BUDGET, RESPONSE_BUDGET, 
    RANDOM_SEED, TEST_SPLIT_SIZE, NUM_LABELS
)

def get_strategic_truncate_processor(tokenizer):
    """
    전략적 잘림(Prompt 앞/Response 뒤)을 수행하는
    .map()용 전처리 함수를 반환합니다.
    """
    
    # 템플릿 토큰화 (미리 계산)
    prompt_template = tokenizer("\nprompt: ", add_special_tokens=False)["input_ids"]
    a_template = tokenizer("\n\nresponse_a: ", add_special_tokens=False)["input_ids"]
    b_template = tokenizer("\n\nresponse_b: ", add_special_tokens=False)["input_ids"]

    def preprocess_function_strategic_truncate(examples):
        # (필수) .map()의 새 프로세스를 위해 numpy 임포트
        import numpy as np 
        
        final_input_ids = []
        final_attention_mask = []
        final_labels = []

        for prompt, response_a, response_b, win_a, win_b, win_tie in zip(
            examples["prompt"],
            examples["response_a"],
            examples["response_b"],
            examples["winner_model_a"],
            examples["winner_model_b"],
            examples["winner_tie"],
        ):
            if win_a == 1: label = 0
            elif win_b == 1: label = 1
            elif win_tie == 1: label = 2
            else: continue 

            prompt_tokens = tokenizer(str(prompt), add_special_tokens=False)["input_ids"][:PROMPT_BUDGET] 
            a_tokens = tokenizer(str(response_a), add_special_tokens=False)["input_ids"][-RESPONSE_BUDGET:]
            b_tokens = tokenizer(str(response_b), add_special_tokens=False)["input_ids"][-RESPONSE_BUDGET:]

            input_ids = (
                [tokenizer.cls_token_id] + DRAFT +
                prompt_template + prompt_tokens + 
                a_template + a_tokens + 
                b_template + b_tokens + 
                [tokenizer.sep_token_id]
            )
            
            padding_length = MAX_LEN - len(input_ids)
            
            if padding_length < 0:
                input_ids = input_ids[:MAX_LEN]
                attention_mask = [1] * MAX_LEN
            else:
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                attention_mask = ([1] * (MAX_LEN - padding_length)) + ([0] * padding_length)
                
            final_input_ids.append(input_ids)
            final_attention_mask.append(attention_mask)
            final_labels.append(label)

        return {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask,
            "labels": final_labels,
        }
    
    # 생성된 함수를 반환
    return preprocess_function_strategic_truncate


def get_dataset_splits(raw_dataset, processor):
    """
    raw_dataset에 processor를 적용하고 train/validation 스플릿을 반환합니다.
    """
    print("Applying preprocessing (strategic truncate, max_length=512)...")
    tokenized_dataset = raw_dataset.map(
        processor,
        batched=True,
        num_proc=1,  # Windows/Jupyter 충돌 방지
        remove_columns=raw_dataset["train"].column_names,
    )
    print("Preprocessing complete.")

    tokenized_dataset = tokenized_dataset["train"].cast_column(
        "labels", ClassLabel(num_classes=NUM_LABELS)
    )
    
    final_datasets = tokenized_dataset.train_test_split(
        test_size=TEST_SPLIT_SIZE, 
        stratify_by_column="labels", 
        seed=RANDOM_SEED
    )
    
    train_dataset = final_datasets["train"]
    val_dataset = final_datasets["test"]
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset

# (분석용) 원본 텍스트가 포함된 val_dataset을 재현하기 위한 함수
def recreate_val_texts_dataset(raw_dataset):
    """
    오류 분석을 위해 원본 텍스트가 포함된 val_dataset을 재생성합니다.
    (훈련 시 사용한 split과 동일한 로직을 사용)
    """
    print("Re-creating validation set with original texts...")
    
    def add_labels_only(example):
        if example["winner_model_a"] == 1: example["labels"] = 0
        elif example["winner_model_b"] == 1: example["labels"] = 1
        elif example["winner_tie"] == 1: example["labels"] = 2
        else: example["labels"] = -1
        return example

    dataset_with_labels = raw_dataset["train"].map(add_labels_only, num_proc=1)
    
    filtered_dataset = dataset_with_labels.filter(
        lambda example: example["labels"] != -1, num_proc=1
    )
    
    casted_dataset = filtered_dataset.cast_column(
        "labels", ClassLabel(num_classes=NUM_LABELS)
    )
    
    temp_final_datasets = casted_dataset.train_test_split(
        test_size=TEST_SPLIT_SIZE, 
        stratify_by_column="labels", 
        seed=RANDOM_SEED
    )
    
    print("Validation texts recreated.")
    return temp_final_datasets["test"]