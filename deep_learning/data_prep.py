import numpy as np
import pandas as pd

"""
入試データ（CSV）を
① 数値ベクトルに変換し
② スケールを揃え
③ 学習用とテスト用に分け
④ NN に渡す (features, targets) を作る

入試データを 表（DataFrame） として読み込む

列：
admit（合否：0/1）
gre
gpa
rank（1〜4のカテゴリ）

one-hotエンコーディング
rank = 2の場合
rank_1 rank_2 rank_3 rank_4
  0      1      0      0
"""

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank (one-hot encoding)
# 数値として扱えない型を数値として扱えるようにする
data = pd.get_dummies(admissions, prefix='rank', columns=['rank'])

# Standardize features
for field in ['gre', 'gpa']:
    data[field] = data[field].astype(np.float64)
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.loc[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']