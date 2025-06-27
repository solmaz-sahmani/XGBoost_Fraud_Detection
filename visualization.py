#!/usr/bin/env python3
"""
ابزارهای نمایش نتایج و رسم نمودارها
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from config import PLOT_CONFIG
import numpy as np

class ResultVisualizer:
    """
    کلاس نمایش و تجسم نتایج
    """
    
    def __init__(self, results):
        self.results = results
    
    def display_results_table(self):
        """
        نمایش جدول نتایج
        """
        print("\n" + "="*80)
        print("نتایج مقایسه مدل‌ها")
        print("="*80)
        
        # ایجاد جدول نتایج
        results_df = pd.DataFrame(self.results).T
        display_columns = ['precision', 'recall', 'f1_score', 'auc', 'training_time']
        results_df = results_df[display_columns].round(4)
        
        print(results_df.to_string())
        
        # بهترین مدل
        best_model = results_df['f1_score'].idxmax()
        best_score = results_df.loc[best_model, 'f1_score']
        
        print(f"\nبهترین مدل بر اساس F1-Score: {best_model}")
        print(f"F1-Score: {best_score:.4f}")
        
        return results_df
    
    def plot_comparison(self):
        """
        رسم نمودار مقایسه معیارهای مختلف
        """
        metrics = PLOT_CONFIG['metrics']
        models = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG['figsize'])
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            bars = axes[i].bar(range(len(models)), values, alpha=0.7)
            
            # تنظیمات نمودار
            axes[i].set_title(f'مقایسه {metric.upper()}', fontsize=12, fontweight='bold')
            axes[i].set_xticks(range(len(models)))
            axes[i].set_xticklabels(models, rotation=PLOT_CONFIG['rotation'], ha='right')
            axes[i].set_ylim(0, 1.1)
            axes[i].grid(True, alpha=0.3)
            
            # افزودن مقادیر روی میله‌ها
            for j, (bar, value) in enumerate(zip(bars, values)):
                axes[i].text(j, value + 0.02, f'{value:.3f}', 
                           ha='center', va='bottom', fontweight='bold')
            
            # رنگ‌آمیزی میله بهترین مدل
            best_idx = np.argmax(values)
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(0.8)
        
        plt.tight_layout()
        plt.show()
    
    def show_confusion_matrices(self, y_test):
        """
        نمایش ماتریس درهم‌ریختگی برای تمام مدل‌ها
        """
        print("\n" + "="*50)
        print("ماتریس درهم‌ریختگی")
        print("="*50)
        
        for model_name in self.results.keys():
            y_pred = self.results[model_name]['y_pred']
            cm = confusion_matrix(y_test, y_pred)
            
            print(f"\n{model_name}:")
            print(f"True Positive (TP): {cm[1,1]}")
            print(f"True Negative (TN): {cm[0,0]}")
            print(f"False Positive (FP): {cm[0,1]}")
            print(f"False Negative (FN): {cm[1,0]}")
            
            # محاسبه نرخ تشخیص و نرخ اخطار کاذب
            if cm[1,1] + cm[1,0] > 0:
                detection_rate = cm[1,1] / (cm[1,1] + cm[1,0])
                print(f"نرخ تشخیص: {detection_rate:.4f}")
            
            if cm[0,1] + cm[0,0] > 0:
                false_alarm_rate = cm[0,1] / (cm[0,1] + cm[0,0])
                print(f"نرخ اخطار کاذب: {false_alarm_rate:.4f}")
    
    def plot_confusion_matrices(self, y_test):
        """
        رسم نمودار ماتریس درهم‌ریختگی
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if i >= len(axes):
                break
                
            y_pred = results['y_pred']
            cm = confusion_matrix(y_test, y_pred)
            
            # رسم ماتریس
            im = axes[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[i].set_title(f'{model_name}')
            
            # افزودن مقادیر
            for row in range(2):
                for col in range(2):
                    axes[i].text(col, row, str(cm[row, col]),
                               ha='center', va='center', fontsize=12, fontweight='bold')
            
            axes[i].set_xlabel('پیش‌بینی')
            axes[i].set_ylabel('واقعی')
            axes[i].set_xticks([0, 1])
            axes[i].set_yticks([0, 1])
            axes[i].set_xticklabels(['عادی', 'کلاهبرداری'])
            axes[i].set_yticklabels(['عادی', 'کلاهبرداری'])
        
        # حذف محورهای اضافی
        for j in range(i + 1, len(axes)):
            axes[j].remove()
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, y_test):
        """
        تولید گزارش کامل
        """
        print("\n" + "="*60)
        print("گزارش کامل نتایج")
        print("="*60)
        
        # نمایش جدول
        results_df = self.display_results_table()
        
        # نمایش ماتریس درهم‌ریختگی
        self.show_confusion_matrices(y_test)
        
        # رسم نمودارها
        self.plot_comparison()
        self.plot_confusion_matrices(y_test)
        
        return results_df