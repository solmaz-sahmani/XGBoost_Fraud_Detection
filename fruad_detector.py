#!/usr/bin/env python3
"""
کلاس اصلی تشخیص کلاهبرداری کارت اعتباری
"""

from data_utils import DataManager
from models import ModelTrainer
from visualization import ResultVisualizer

class FraudDetector:
    """
    کلاس اصلی سیستم تشخیص کلاهبرداری
    """
    
    def __init__(self):
        self.data_manager = DataManager()
        self.model_trainer = ModelTrainer()
        self.visualizer = None
        self.results = None
    
    def run_complete_analysis(self, data_path=None):
        """
        اجرای تحلیل کامل
        """
        print("سیستم تشخیص کلاهبرداری کارت اعتباری")
        print("="*50)
        
        # مرحله 1: بارگذاری داده‌ها
        print("\n1. بارگذاری داده‌ها...")
        df = self.data_manager.load_data(data_path)
        
        # مرحله 2: پیش‌پردازش
        print("\n2. پیش‌پردازش داده‌ها...")
        X_train, X_test, y_train, y_test = self.data_manager.preprocess_data(df)
        
        # مرحله 3: آموزش مدل‌ها
        print("\n3. آموزش مدل‌ها...")
        self.results = self.model_trainer.train_models(X_train, X_test, y_train, y_test)
        
        # مرحله 4: نمایش نتایج
        print("\n4. نمایش نتایج...")
        self.visualizer = ResultVisualizer(self.results)
        results_df = self.visualizer.generate_report(y_test)
        
        print("\nتحلیل کامل شد!")
        return results_df
    
    def predict_new_data(self, new_data, model_name='Improved XGBoost'):
        """
        پیش‌بینی برای داده‌های جدید
        """
        if self.model_trainer.models is None or model_name not in self.model_trainer.models:
            print("ابتدا مدل‌ها را آموزش دهید!")
            return None
        
        # استاندارد کردن داده‌های جدید
        new_data_scaled = self.data_manager.scaler.transform(new_data)
        
        # پیش‌بینی
        model = self.model_trainer.models[model_name]
        predictions = model.predict(new_data_scaled)
        probabilities = model.predict_proba(new_data_scaled)
        
        return predictions, probabilities
    
    def get_model_performance(self, model_name):
        """
        دریافت عملکرد یک مدل خاص
        """
        if self.results is None or model_name not in self.results:
            return None
        
        return self.results[model_name]
    
    def compare_models(self, metric='f1_score'):
        """
        مقایسه مدل‌ها بر اساس معیار مشخص
        """
        if self.results is None:
            print("ابتدا مدل‌ها را آموزش دهید!")
            return None
        
        model_scores = {}
        for model_name, results in self.results.items():
            model_scores[model_name] = results[metric]
        
        # مرتب‌سازی بر اساس امتیاز
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nرتبه‌بندی مدل‌ها بر اساس {metric}:")
        print("-" * 40)
        for i, (model, score) in enumerate(sorted_models, 1):
            print(f"{i}. {model}: {score:.4f}")
        
        return sorted_models
    
    def save_results(self, filename='fraud_detection_results.csv'):
        """
        ذخیره نتایج در فایل
        """
        if self.results is None:
            print("هیچ نتیجه‌ای برای ذخیره وجود ندارد!")
            return
        
        import pandas as pd
        
        # تبدیل نتایج به DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df[['precision', 'recall', 'f1_score', 'auc', 'training_time']]
        
        # ذخیره در فایل
        results_df.to_csv(filename)
        print(f"نتایج در فایل {filename} ذخیره شد.")
    
    def load_pretrained_model(self, model_path):
        """
        بارگذاری مدل از پیش آموزش دیده
        """
        import pickle
        
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.model_trainer.models = saved_data['models']
            self.data_manager.scaler = saved_data['scaler']
            print("مدل با موفقیت بارگذاری شد!")
            
        except Exception as e:
            print(f"خطا در بارگذاری مدل: {e}")
    
    def save_trained_models(self, filename='trained_models.pkl'):
        """
        ذخیره مدل‌های آموزش دیده
        """
        if self.model_trainer.models is None:
            print("هیچ مدلی برای ذخیره وجود ندارد!")
            return
        
        import pickle
        
        save_data = {
            'models': self.model_trainer.models,
            'scaler': self.data_manager.scaler,
            'results': self.results
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"مدل‌ها در فایل {filename} ذخیره شدند.")
            
        except Exception as e:
            print(f"خطا در ذخیره مدل‌ها: {e}")