# model_trainer.py

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix

# --- 설정 변수 ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def clean_prompt(s):
    """프롬프트에서 불필요한 구두점 제거 (임베딩 개선 목적)"""
    return re.sub(r'["\'\[\]]', '', str(s))

def generate_embeddings(df, model):
    """
    Prompt + Response 텍스트를 결합하고 MiniLM 임베딩을 추출합니다.
    """
    print(f"➡️ Generating embeddings using {EMBEDDING_MODEL_NAME}...")
    
    df['prompt_clean'] = df['prompt'].apply(clean_prompt)
    df['text_a'] = df['prompt_clean'] + ' ' + df['response_a']
    df['text_b'] = df['prompt_clean'] + ' ' + df['response_b']
    
    # Prompt, Response A, Response B 임베딩 추출
    emb_a = model.encode(df['text_a'].tolist(), show_progress_bar=True)
    emb_b = model.encode(df['text_b'].tolist(), show_progress_bar=True)
    prompt_emb = model.encode(df['prompt_clean'].tolist(), show_progress_bar=True)
    
    # 응답 간 차이 벡터 (A - B)
    emb_diff = emb_a - emb_b
    
    return emb_diff, prompt_emb

def build_hybrid_data(emb_diff, prompt_emb, X_train_scaled):
    """
    임베딩 벡터와 스케일링된 테이블 특징을 결합하여 최종 X 데이터를 생성합니다.
    """
    print("➡️ Combining embeddings and tabular features...")
    # X_train_scaled train_df의 인덱스 순서와 동일해야 함
    
    # NOTE: 테스트 데이터가 들어올 경우, X_train_scaled는 테스트 데이터의 특징만 포함해야 합니다.
    # 이 함수는 train/test 별도로 호출하는 것이 안전합니다.
    
    X_hybrid = np.concatenate([emb_diff, prompt_emb, X_train_scaled.values], axis=1)
    print(f"   Hybrid X shape: {X_hybrid.shape}")
    return X_hybrid

def train_and_evaluate_model(X_train, y_train, X_val, y_val, search_params=True):
    """
    GridSearchCV와 StackingClassifier를 사용하여 모델을 학습하고 검증합니다.
    """
    print("➡️ Initializing Stacking Classifier...")
    
    # 1. Base models 정의
    base_models = [
        ('lr', LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)),
        ('lgb', LGBMClassifier(random_state=42, verbose=-1))]
    
    # 2. Meta model 정의
    meta_model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', random_state=42)
    
    # 3. StackingClassifier 정의
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        stack_method='predict_proba',
        cv=3,       # 내부 Stacking CV
        n_jobs=-1)
        
    if search_params:
        print("➡️ Performing GridSearchCV for Hyperparameter Tuning (may take time)...")
        # 4. GridSearch를 위한 파라미터 그리드 설정 (원본 코드 기반)
        param_grid = {
            'lgb__n_estimators': [200, 400],
            'lgb__learning_rate': [0.03, 0.05],
            'rf__n_estimators': [100, 200]}

        # 5. GridSearchCV 감싸기
        grid = GridSearchCV(
            estimator=stacking_model,
            param_grid=param_grid,
            scoring='neg_log_loss', # Log Loss 최소화
            cv=3,                   # 전체 모델에 대한 CV
            verbose=1,
            n_jobs=-1)
            
        # 6. 학습 수행
        grid.fit(X_train, y_train)
        
        # 7. 결과 출력
        print("\n🏆 Best Model Found:")
        print(f"   Best Parameters: {grid.best_params_}")
        print(f"   Best CV LogLoss: {-grid.best_score_:.4f}")
        
        final_model = grid.best_estimator_
    else:
        # 간단 학습 (디버깅용)
        print("➡️ Training Stacking Model without GridSearch...")
        final_model = stacking_model
        final_model.fit(X_train, y_train)

    # 8. 검증 및 평가
    y_pred = final_model.predict(X_val)
    y_prob = final_model.predict_proba(X_val)
    
    acc = final_model.score(X_val, y_val)
    loss = log_loss(y_val, y_prob)
    
    print(f"\n📊 Validation Accuracy: {acc:.4f}")
    print(f"📉 Validation LogLoss: {loss:.4f}")
    
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, y_pred, target_names=['A win', 'B win', 'Tie']))
    
    # 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=['A win', 'B win', 'Tie'], yticklabels=['A win', 'B win', 'Tie'])
    plt.title("Confusion Matrix (Validation)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    plt.subplot(1, 2, 2)
    try:
        meta = final_model.final_estimator_
        # Meta Model Coefficients (LR의 경우)
        coef_df = pd.DataFrame(meta.coef_.T, 
                               index=[name + "_prob" for name, _ in base_models], 
                               columns=['A win', 'B win', 'Tie'])
        coef_df.plot(kind='bar', ax=plt.gca())
        plt.title("Meta Model Feature Importance (LR Coefs)")
        plt.ylabel("Coefficient Value")
        plt.xticks(rotation=45, ha='right')
    except Exception as e:
        print(f"Could not plot meta model coefficients: {e}")

    plt.tight_layout()
    plt.show()

    return final_model, y_prob