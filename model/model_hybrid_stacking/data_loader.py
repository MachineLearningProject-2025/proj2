# data_loader.py

import pandas as pd
import numpy as np

def load_data(data_dir='.../dataset'):
    """
    Train, Test, Submission 파일을 로드하고 Target 변수 (Label)를 생성합니다.
    """
    print("➡️ Loading data...")
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    submission_df = pd.read_csv(f'{data_dir}/sample_submission.csv')
    
    # Target (Label) 변수 생성: A win=0, B win=1, Tie=2
    train_df['label'] = np.select(
        [train_df['winner_model_a']==1, train_df['winner_model_b']==1, train_df['winner_tie']==1],
        [0, 1, 2],
        default=-1
    )
    
    print(f"   Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    return train_df, test_df, submission_df

# 메인 스크립트에서 import 할 때 실행되지 않도록 방지
if __name__ == '__main__':
    train, test, sub = load_data()
    print("\nTrain Head:")
    print(train[['winner_model_a', 'winner_model_b', 'winner_tie', 'label']].head())