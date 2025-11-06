# config.py
# --------------------------------------------------
# 이 파일은 훈련과 분석 스크립트가 공유하는 모든 설정을 저장합니다.
# --------------------------------------------------

import torch

# --- 1. 경로 설정 ---
# (환경에 맞게 수정하세요)
DATA_PATH = "../dataset/" 
# DATA_PATH = "/kaggle/input/llm-classification-finetuning/"

# 훈련된 모델과 아티팩트를 저장할 경로
OUTPUT_DIR = "./results_lora_strategic"
LOGGING_DIR = "./logs_lora_strategic"

OUTPUT_SUBMISSION_FILE = 'submission.csv'

# --- 2. 모델 설정 ---
BASE_MODEL_NAME = "microsoft/deberta-v3-small"
# BASE_MODEL_PATH = "/kaggle/input/deberta-v3-small/..." # (Kaggle용 로컬 경로)

# --- 3. 토큰화 "예산" 설정 ---
# (전략적 잘림에 사용)
MAX_LEN = 512
PROMPT_BUDGET = 150
RESPONSE_BUDGET = 170

# --- 4. 훈련 하이퍼파라미터 ---
NUM_LABELS = 3
TRAIN_EPOCHS = 2
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
EVAL_STEPS = 5000
SAVE_STEPS = 5000

# --- 5. LoRA 설정 ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query_proj", "value_proj"]
LORA_CHECKPOINT_PATH = "./lora_strategic"

# --- 6. 환경 및 분할 설정 ---
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.2
LABELS_MAP = ["A_win (0)", "B_win (1)", "Tie (2)"]
NUM_LABELS = len(LABELS_MAP)