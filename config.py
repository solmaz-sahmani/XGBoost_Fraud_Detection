#!/usr/bin/env python3

#config
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_DATA_SIZE = 50000
FRAUD_RATE = 0.002  # 0.2% fraud

# SMOTE config
SMOTE_CONFIG = {
    'random_state': RANDOM_STATE
}

# Models config
MODEL_CONFIGS = {
    'Logistic Regression': {
        'random_state': RANDOM_STATE,
        'max_iter': 1000
    },
    'Decision Tree': {
        'random_state': RANDOM_STATE
    },
    'Random Forest': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE
    },
    'XGBoost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    },
    'Improved XGBoost': {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_STATE
    }
}

# diagram config
PLOT_CONFIG = {
    'figsize': (15, 10),
    'rotation': 45,
    'metrics': ['precision', 'recall', 'f1_score', 'auc']
}