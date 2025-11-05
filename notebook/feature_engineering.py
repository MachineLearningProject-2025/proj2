# feature_engineering.py

import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer

# --- 설정 변수 ---
# 최종 모델에서 사용된 유의미한 차이 특징 및 키워드
SIGNIFICANT_FEATURES = [
    'len_diff', 'punc_diff', 'sent_count_diff', 
    'lexical_div_diff', 'comma_ratio_diff',
]
TARGET_WORDS = [
    'company', 'brace', 'knee', 'progression', 
    'apologize', 'sorry'
]
KEYWORD_FEATURES = [f'contains_{w}' for w in TARGET_WORDS]
ALL_FEATURES = SIGNIFICANT_FEATURES + KEYWORD_FEATURES

def _calculate_linguistic_features(text):
    """단일 텍스트에 대해 어휘 다양성, 반복률, 주관성을 계산합니다."""
    words = re.findall(r'\b\w+\b', str(text).lower())
    len_words = len(words)
    if len_words == 0:
        return 0.0, 0.0, 0.0
    
    uniq = set(words)
    lex_div = len(uniq) / len_words
    repetition = 1 - lex_div
    blob = TextBlob(str(text))
    subj = blob.sentiment.subjectivity
    return lex_div, repetition, subj

def extract_features(df):
    """
    데이터프레임에 언어학적/통계적 특징을 추출하고 차이(diff) 특징을 추가합니다.
    """
    print("➡️ Extracting lexical and linguistic features...")
    
    for side in ['a', 'b']:
        response_col = f'response_{side}'
        
        # 1. 길이 및 구두점 특징
        df[f'len_{side}'] = df[response_col].str.len()
        df[f'punc_{side}'] = df[response_col].apply(lambda x: len(re.findall(r'[!?,;:]', str(x))))
        df[f'sent_{side}'] = df[response_col].apply(lambda x: len(re.findall(r'[.!?]', str(x))))

        # 2. 언어학적 특징 (TextBlob 기반)
        df[[f'lex_{side}', f'rep_{side}', f'subj_{side}']] = df[response_col].apply(
            lambda x: pd.Series(_calculate_linguistic_features(x))
        )
        # 3. 콤마 비율
        df[f'comma_{side}'] = df[response_col].apply(
            lambda x: len(re.findall(r'[;,]', str(x))) / (len(str(x).split()) + 1e-9)
        )

    # 4. 차이(Diff) 특징 계산
    df['len_diff'] = df['len_a'] - df['len_b']
    df['punc_diff'] = df['punc_a'] - df['punc_b']
    df['sent_count_diff'] = df['sent_a'] - df['sent_b']
    df['lexical_div_diff'] = df['lex_a'] - df['lex_b']
    df['comma_ratio_diff'] = df['comma_a'] - df['comma_b']
    
    # 5. 키워드 존재 유무
    # prompt + response_a + response_b 합쳐서 검색
    text_cols = df[['prompt', 'response_a', 'response_b']].astype(str).agg(' '.join, axis=1)
    for word in TARGET_WORDS:
        df[f'contains_{word}'] = text_cols.str.contains(fr'\b{word}\b', case=False, na=False).astype(int)

    return df

def scale_features(X_df):
    """
    특징 통계 기반으로 적절한 스케일링(RobustScaler, StandardScaler, Passthrough)을 적용합니다.
    """
    print("➡️ Scaling features (Robust/Standard/Passthrough)...")
    
    df = X_df.copy()
    
    # 통계 기반 스케일러 추천
    summary = []
    for col in df.columns:
        vals = df[col].dropna().astype(float)
        is_binary = set(vals.unique()).issubset({0.0, 1.0}) or df[col].nunique(dropna=True) == 2
        
        # 스케일러 권장 로직 (원본 코드 참고)
        recommended = 'passthrough (binary)' if is_binary else (
            'RobustScaler' if (vals.var() < 1e-4) or (abs(vals.skew()) > 1.0) or 
            ( (vals.quantile(0.75) - vals.quantile(0.25)) > 0 and 
              (((vals < (vals.quantile(0.25) - 1.5 * (vals.quantile(0.75) - vals.quantile(0.25)))) | 
               (vals > (vals.quantile(0.75) + 1.5 * (vals.quantile(0.75) - vals.quantile(0.25))))).sum() / max(len(vals), 1) > 0.05)
            ) else 'StandardScaler'
        )
        summary.append({'feature': col, 'is_binary': is_binary, 'recommended': recommended})
    
    summary_df = pd.DataFrame(summary).set_index('feature')
    
    # 그룹 분류
    binary_cols = summary_df[summary_df['is_binary']].index.tolist()
    robust_cols = summary_df[summary_df['recommended'] == 'RobustScaler'].index.difference(binary_cols).tolist()
    std_cols = summary_df[summary_df['recommended'] == 'StandardScaler'].index.difference(binary_cols).tolist()
    
    # ColumnTransformer 구성
    transformers = []
    if std_cols:
        transformers.append(('std', StandardScaler(), std_cols))
    if robust_cols:
        transformers.append(('robust', RobustScaler(), robust_cols))
    if binary_cols:
        transformers.append(('passthrough_binary', 'passthrough', binary_cols))

    if not transformers:
        print("   Warning: No features to transform.")
        return X_df
        
    col_transformer = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)
    
    # Fit & Transform
    X_scaled_arr = col_transformer.fit_transform(df)
    
    # 컬럼명 재구성
    out_cols = []
    for name, _, cols in transformers:
        out_cols.extend(cols if isinstance(cols, (list, tuple)) else list(cols))
        
    X_scaled_df = pd.DataFrame(X_scaled_arr, columns=out_cols, index=df.index)
    print(f"   Scaled X_df shape: {X_scaled_df.shape}")
    return X_scaled_df, col_transformer

# 메인 스크립트에서 import 할 때 실행되지 않도록 방지
if __name__ == '__main__':
    from data_loader import load_data
    train_df, test_df, _ = load_data()
    
    train_df = extract_features(train_df)
    
    X_tab = train_df[ALL_FEATURES]
    X_scaled, _ = scale_features(X_tab)
    
    print("\nScaled Features Head:")
    print(X_scaled.head())