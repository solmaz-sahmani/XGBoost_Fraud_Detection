#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from config import *

class DataManager:
    """
    data manage class
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(**SMOTE_CONFIG)
    
    def load_data(self, data_path=None):
        """
        loading data
        """
        if data_path:
            df = pd.read_csv(data_path)
        else:
            df = self.generate_sample_data()
        
        print(f"تعداد کل داده‌ها: {len(df)}")
        print(f"تعداد کلاهبرداری: {df['Class'].sum()}")
        print(f"درصد کلاهبرداری: {df['Class'].mean()*100:.2f}%")
        
        return df
    
    def preprocess_data(self, df):
        """
        data preprocessing, Standardizing, Applying SMOTE
        """
        # Separating features and labels
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Standardizing the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Applying SMOTE to balance the data
        print("اعمال SMOTE برای متعادل کردن داده‌ها...")
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train_scaled, y_train)
        
        print(f"بعد از SMOTE - تعداد داده‌های آموزشی: {len(X_train_balanced)}")
        print(f"بعد از SMOTE - تعداد کلاهبرداری: {y_train_balanced.sum()}")
        
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test
    
    def generate_sample_data(self, n_samples=SAMPLE_DATA_SIZE):
        """
        Generating sample data for testing
        """
        np.random.seed(RANDOM_STATE)
        
        # Number of fraud cases
        n_fraud = int(n_samples * FRAUD_RATE)
        n_normal = n_samples - n_fraud
        
        # Generating features (simulating V1–V28)
        n_features = 28
        
        # Normal transactions
        normal_data = np.random.normal(0, 1, (n_normal, n_features))
        normal_labels = np.zeros(n_normal)
        
        # Fraudulent transactions
        fraud_data = np.random.normal(1, 1.5, (n_fraud, n_features))
        fraud_labels = np.ones(n_fraud)
        
        # Data merging
        X = np.vstack([normal_data, fraud_data])
        y = np.hstack([normal_labels, fraud_labels])
        
        # Adding the Time and Amount columns
        times = np.random.uniform(0, 172800, n_samples)
        amounts = np.random.lognormal(3, 1.5, n_samples)
        
        # Creating a DataFrame
        columns = [f'V{i}' for i in range(1, n_features+1)] + ['Time', 'Amount']
        data = np.column_stack([X, times, amounts])
        
        df = pd.DataFrame(data, columns=columns)
        df['Class'] = y.astype(int)
        
        return df