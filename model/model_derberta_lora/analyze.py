# analyze.py
# --------------------------------------------------
# STAGE 2: 훈련된 모델 로드, 검증 세트 분석, 보정기 훈련
# --------------------------------------------------

import os
import datasets
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib # 보정기 저장을 위해
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    log_loss,
    accuracy_score
)
from sklearn.isotonic import IsotonicRegression

# 설정 및 유틸리티 함수 임포트
from config import *
from utils import get_strategic_truncate_processor, get_dataset_splits, recreate_val_texts_dataset

def main_analyze():
    print("🚀 STAGE 2: Starting Model Analysis...")
    
    # 0. (필수) 병렬 처리 비활성화
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.disable_multiprocessing()

    # 1. 훈련된 모델과 토크나이저 로드
    print(f"Loading trained model and tokenizer from {OUTPUT_DIR}...")
    try:
        # (주의) 훈련 시 사용한 베이스 모델을 먼저 로드
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, num_labels=NUM_LABELS, device_map="auto"
        )
        # LoRA 어댑터(저장된 모델)를 덮어씌움
        lora_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    except Exception as e:
        print(f"훈련된 모델 로드 실패: {e}")
        print(f"{OUTPUT_DIR} 경로에 훈련된 모델이 있는지 확인하세요.")
        return
    print("Model and tokenizer loaded.")
    lora_model.eval() # 추론 모드로 설정

    # 2. 분석을 위한 데이터 재생성
    # (훈련 시와 동일한 전처리/스플릿을 수행하여 val_dataset을 재현)
    try:
        raw_dataset = load_dataset("csv", data_files=DATA_PATH + "train.csv")
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return

    processor = get_strategic_truncate_processor(tokenizer)
    _, val_dataset = get_dataset_splits(raw_dataset, processor)

    # 3. 예측 실행
    print("Running predictions on validation set...")
    # (W&B 로깅을 끄기 위해 dummy_args 사용)
    dummy_args = TrainingArguments(output_dir="./dummy_results", report_to="none")
    trainer = Trainer(model=lora_model, args=dummy_args)
    
    predictions_output = trainer.predict(val_dataset)
    
    # 4. 예측 결과 추출
    logits = predictions_output.predictions
    y_true = predictions_output.label_ids
    y_probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
    y_pred = np.argmax(y_probs, axis=1)

    # 5. 보정 전(Uncalibrated) 분석
    print("\n--- 📊 1. Uncalibrated Analysis ---")
    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, y_probs)
    print(f"Validation Accuracy (Uncalibrated): {acc:.4f}")
    print(f"Validation LogLoss (Uncalibrated): {loss:.4f}")

    print("Plotting Confusion Matrix (Uncalibrated)...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS_MAP)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Uncalibrated (Validation)")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_uncalibrated.png"))
    plt.show()

    print("\nClassification Report (Uncalibrated):\n")
    print(classification_report(y_true, y_pred, target_names=LABELS_MAP))

    # 6. 오류 분석 (Error Inspection)
    print("\n--- 😱 2. Error Analysis ---")
    # 원본 텍스트가 포함된 val_dataset 재생성
    val_texts_dataset = recreate_val_texts_dataset(raw_dataset)
    
    df_report = pd.DataFrame(val_texts_dataset)
    df_report["pred_label"] = y_pred
    df_report["true_label"] = y_true

    df_errors = df_report[df_report["true_label"] != df_report["pred_label"]]
    print(f"Total prediction errors: {len(df_errors)}")
    
    # 오류 상위 10개 출력
    label_names_dict = {i: name for i, name in enumerate(LABELS_MAP)}
    for index, row in df_errors.head(10).iterrows():
        print("=" * 40)
        print(f"    👉 정답 (True): {label_names_dict[row['true_label']]}")
        print(f"    👉 예측 (Pred): {label_names_dict[row['pred_label']]}")
        print("-" * 40)
        print(f"[Prompt]: {row['prompt'][:200]}...") # 너무 길지 않게
        print(f"[Response A]: {row['response_a'][:200]}...")
        print(f"[Response B]: {row['response_b'][:200]}...")
        print("-" * 40 + "\n")
    
    # 오류 전체 CSV로 저장
    errors_csv_path = os.path.join(OUTPUT_DIR, "prediction_errors.csv")
    df_errors.to_csv(errors_csv_path, index=False, encoding='utf-8-sig')
    print(f"All prediction errors saved to: {errors_csv_path}")

    # 7. 보정기 훈련 및 저장
    print("\n--- 📈 3. Calibration Training ---")
    calibrators = {}
    for i in range(NUM_LABELS):
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        y_cal = (y_true == i).astype(int)
        iso_reg.fit(y_probs[:, i], y_cal)
        calibrators[i] = iso_reg

    print("Calibration models trained.")
    
    # 보정기 저장
    calibrator_path = os.path.join(OUTPUT_DIR, "calibrators.joblib")
    joblib.dump(calibrators, calibrator_path)
    print(f"Calibrators saved to: {calibrator_path}")
    
    print("\n✅ STAGE 2: Analysis and calibration complete.")

if __name__ == "__main__":
    main_analyze()