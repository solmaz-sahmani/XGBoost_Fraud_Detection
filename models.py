#!/usr/bin/env python3
"""
مدل‌های یادگیری ماشین برای تشخیص کلاهبرداری
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix)
import time
from config import MODEL_CONFIGS

class ModelTrainer:
    """
    کلاس آموزش و ارزیابی مدل‌ها
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def get_models(self):
        """
        ایجاد مدل‌های مختلف
        """
        models = {
            'Logistic Regression': LogisticRegression(**MODEL_CONFIGS['Logistic Regression']),
            'Decision Tree': DecisionTreeClassifier(**MODEL_CONFIGS['Decision Tree']),
            'Random Forest': RandomForestClassifier(**MODEL_CONFIGS['Random Forest']),
            'XGBoost': xgb.XGBClassifier(**MODEL_CONFIGS['XGBoost']),
            'Improved XGBoost': xgb.XGBClassifier(**MODEL_CONFIGS['Improved XGBoost'])
        }
        return models
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        آموزش تمام مدل‌ها و ارزیابی آنها
        """
        models = self.get_models()
        
        print("شروع آموزش مدل‌ها...")
        for name, model in models.items():
            print(f"آموزش {name}...")
            
            # اندازه‌گیری زمان آموزش
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # پیش‌بینی
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # محاسبه معیارها
            results = self.evaluate_model(y_test, y_pred, y_pred_proba, training_time)
            
            # ذخیره نتایج
            self.results[name] = results
            self.models[name] = model
            
            print(f"{name} کامل شد - زمان: {training_time:.2f} ثانیه")
        
        return self.results
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba, training_time):
        """
        ارزیابی یک مدل
        """
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'training_time': training_time,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def get_confusion_matrix(self, model_name, y_test):
        """
        محاسبه ماتریس درهم‌ریختگی برای یک مدل
        """
        if model_name not in self.results:
            return None
        
        y_pred = self.results[model_name]['y_pred']
        return confusion_matrix(y_test, y_pred)
    
    def get_best_model(self, metric='f1_score'):
        """
        یافتن بهترین مدل بر اساس معیار مشخص
        """
        if not self.results:
            return None
        
        best_score = 0
        best_model = None
        
        for model_name, results in self.results.items():
            if results[metric] > best_score:
                best_score = results[metric]
                best_model = model_name
        
        return best_model, best_score