import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
# Pearson Based Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# Feature Importance Based Feature Selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split # 파라미터 튜닝
from imblearn.pipeline import Pipeline # 전처리, 모델 등을 하나의 pipeline으로 그룹화
from imblearn.over_sampling import SMOTE # Oversampling

# Hybrid Feature Selection 
def label_feature_selection(y, df): # 타겟 데이터가 multi-class인 경우, 중요도에 따라 증강할 타겟 데이터 선별
    X = pd.get_dummies(y) # One-hot encoding
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pearson Based Feature Selection
    k = round(len(y.value_counts().index) * 0.8) # 전체 데이터의 80% 비율.
    pearson_selector = SelectKBest(f_classif, k=k)
    pearson_selector.fit(X_train, y_train)

    # Feature Importance Based Feature Selection
    importance_selector = SelectFromModel(RandomForestClassifier(n_estimators = 100, random_state = 42))     # threshold=0.0001 
    importance_selector.fit(X_train,y_train)

    # Feature 교집합 inner join
    pearson_based_features = pd.DataFrame(pearson_selector.get_support(indices=True))
    importance_based_features = pd.DataFrame(importance_selector.get_support(indices=True))
    inner_df = pd.merge(pearson_based_features, importance_based_features, how='inner')
    intersection_features = [feature for feature in inner_df[0]]
    intersection_df = df[y.isin(intersection_features)] # Hybrid Feature Selection 데이터프레임에 적용
    
    return intersection_df

def make_grid(updated_df, target_column): # 데이터 증강 비율 계산
    max_strategy = {}
    mean_strategy = {}
    median_strategy = {}

    value_series = updated_df[target_column].value_counts()
    labels_values = value_series.values
    label_counts = value_series.items()

    max_value = labels_values.max()
    mean_value = np.mean(labels_values)
    median_value = np.median(labels_values)

    for key, value in label_counts:
                                                                                                       
        max_strategy.update({key: round(max_value)}) 
        if value < mean_value:                                                                          # 빈도수가 평균 이하면, 평균으로 변환
            mean_strategy.update({key: round(mean_value)})
        if value < median_value:                                                                        # 빈도수가 중앙값 이하면, 중앙값으로 변환
            median_strategy.update({key: round(np.median(median_value))})

    pipeline = Pipeline([
        ('oversample', SMOTE(k_neighbors=3)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])

    param_grid = {'oversample__sampling_strategy': [max_strategy, mean_strategy, median_strategy]}
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

    X = updated_df.drop(target_column, axis=1)
    y = updated_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    grid_search.fit(X_train, y_train)

    print("Best hyperparameters: ", grid_search.best_params_)

    return grid_search.best_params_

def make_ratio(sampling_ratio):
    sampling_strategy = {}                                # ex) sampling_strategy = {2: 5344, 8: 5344, 4: 5344, 3: 5344, 6: 5344}
    for best_ratio in sampling_ratio.values():
        sampling_strategy.update(best_ratio)
    return sampling_strategy

