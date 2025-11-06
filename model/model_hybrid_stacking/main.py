# main.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# 커스텀 모듈 임포트
from data_loader import load_data
from feature_engineering import extract_features, scale_features, ALL_FEATURES
from model_trainer import (
    EMBEDDING_MODEL_NAME, 
    generate_embeddings, 
    build_hybrid_data, 
    train_and_evaluate_model
)

def main():
    print("🚀 Starting LLM Preference Prediction Project Workflow...")
    
    # 1. 데이터 로드
    train_df, test_df, submission_df = load_data(data_dir='.../dataset')
    
    # 2. 특징 추출
    train_df = extract_features(train_df)
    test_df = extract_features(test_df)
    
    X_train_tab = train_df[ALL_FEATURES]
    X_test_tab = test_df[ALL_FEATURES]
    y_train = train_df['label'].values
    
    # 3. 테이블 특징 스케일링 (Train 데이터로 fit)
    X_train_scaled, feature_scaler = scale_features(X_train_tab)
    # Test 데이터는 Train 데이터의 fit 결과를 사용하여 transform
    X_test_scaled = feature_scaler.transform(X_test_tab)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train_scaled.columns, index=X_test_tab.index)
    
    # 4. Sentence Transformer 모델 로드
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error loading S-BERT model: {e}. Please check your connection or model name.")
        return
    
    # 5. 임베딩 추출
    emb_diff_train, prompt_emb_train = generate_embeddings(train_df, embed_model)
    emb_diff_test, prompt_emb_test = generate_embeddings(test_df, embed_model)
    
    # 6. 하이브리드 데이터 생성
    X_train_hybrid = build_hybrid_data(emb_diff_train, prompt_emb_train, X_train_scaled)
    X_test_hybrid = build_hybrid_data(emb_diff_test, prompt_emb_test, X_test_scaled_df)
    
    # 7. 학습 및 검증 데이터 분리
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_hybrid, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train)
        
    print(f"\nTraining Split: {X_tr.shape}, Validation Split: {X_val.shape}")

    # 8. 모델 학습 및 평가 (GridSearch 수행)
    final_model, y_val_prob = train_and_evaluate_model(X_tr, y_tr, X_val, y_val, search_params=True)
    
    print("\n✅ Training and Validation Complete.")
    
    # 9. 최종 Test 데이터 예측 및 CSV 파일 생성
    print("\n➡️ Generating final prediction for submission...")
    y_test_prob = final_model.predict_proba(X_test_hybrid)
    
    # Submission 파일 형식 맞추기
    submission_df['score_a'] = y_test_prob[:, 0]
    submission_df['score_b'] = y_test_prob[:, 1]
    submission_df['score_c'] = y_test_prob[:, 2] # Tie
    
    submission_path = '.../submission/final_submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"   Submission file saved to: {submission_path}")

if __name__ == '__main__':
    main()