#!/usr/bin/env python3
"""
اسکریپت اصلی برای اجرای سیستم تشخیص کلاهبرداری کارت اعتباری
"""

import warnings
warnings.filterwarnings('ignore')

from fraud_detector import FraudDetector
import argparse
import sys

def main():
    """
    تابع اصلی برنامه
    """
    # تنظیم آرگومان‌های خط فرمان
    parser = argparse.ArgumentParser(description='سیستم تشخیص کلاهبرداری کارت اعتباری')
    parser.add_argument('--data', type=str, help='مسیر فایل داده (اختیاری)')
    parser.add_argument('--save-results', action='store_true', help='ذخیره نتایج در فایل')
    parser.add_argument('--save-models', action='store_true', help='ذخیره مدل‌های آموزش دیده')
    parser.add_argument('--compare-only', action='store_true', help='فقط مقایسه مدل‌ها')
    
    args = parser.parse_args()
    
    try:
        # ایجاد detector
        detector = FraudDetector()
        
        # اجرای تحلیل کامل
        results_df = detector.run_complete_analysis(args.data)
        
        # ذخیره نتایج در صورت درخواست
        if args.save_results:
            detector.save_results('fraud_detection_results.csv')
        
        # ذخیره مدل‌ها در صورت درخواست
        if args.save_models:
            detector.save_trained_models('trained_models.pkl')
        
        # مقایسه مدل‌ها بر اساس معیارهای مختلف
        if args.compare_only or True:  # همیشه مقایسه نمایش داده شود
            print("\n" + "="*60)
            print("مقایسه تفصیلی مدل‌ها")
            print("="*60)
            
            metrics = ['precision', 'recall', 'f1_score', 'auc']
            for metric in metrics:
                detector.compare_models(metric)
        
        # نمایش پیام پایانی
        print("\n" + "="*60)
        print("اجرای برنامه با موفقیت تمام شد!")
        print("="*60)
        
        # نمایش خلاصه نتایج
        best_model = results_df['f1_score'].idxmax()
        best_f1 = results_df.loc[best_model, 'f1_score']
        print(f"\nبهترین مدل: {best_model}")
        print(f"بهترین F1-Score: {best_f1:.4f}")
        
        return results_df
        
    except KeyboardInterrupt:
        print("\n\nبرنامه توسط کاربر متوقف شد.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nخطایی رخ داد: {e}")
        sys.exit(1)

def demo_prediction():
    """
    نمونه‌ای از پیش‌بینی برای داده‌های جدید
    """
    import numpy as np
    
    print("\nنمونه پیش‌بینی برای داده جدید:")
    print("-" * 40)
    
    # ایجاد detector و آموزش
    detector = FraudDetector()
    detector.run_complete_analysis()
    
    # تولید داده نمونه برای پیش‌بینی
    np.random.seed(123)
    sample_data = np.random.normal(0, 1, (5, 30))  # 5 نمونه با 30 ویژگی
    
    # پیش‌بینی
    predictions, probabilities = detector.predict_new_data(sample_data)
    
    print("نتایج پیش‌بینی:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        fraud_prob = prob[1] * 100
        status = "کلاهبرداری" if pred == 1 else "عادی"
        print(f"نمونه {i+1}: {status} (احتمال کلاهبرداری: {fraud_prob:.2f}%)")

if __name__ == "__main__":
    # اجرای برنامه اصلی
    if len(sys.argv) > 1 and '--demo' in sys.argv:
        demo_prediction()
    else:
        main()