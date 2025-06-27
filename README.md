# سیستم تشخیص کلاهبرداری کارت اعتباری

این پروژه یک سیستم کامل برای تشخیص کلاهبرداری کارت اعتباری با استفاده از تکنیک‌های یادگیری ماشین و SMOTE برای متعادل کردن داده‌ها است.

## ویژگی‌ها

- **مدل‌های مختلف ML**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **تعادل داده‌ها**: استفاده از تکنیک SMOTE
- **ارزیابی کامل**: معیارهای Precision, Recall, F1-Score, AUC
- **تجسم نتایج**: نمودارهای مقایسه و ماتریس درهم‌ریختگی
- **ساختار Modular**: کد تقسیم‌بندی شده در فایل‌های مجزا

## ساختار پروژه

```
XGBoost Fraud Detection/
├── config.py              # تنظیمات پروژه
├── data_utils.py           # مدیریت و پردازش داده‌ها
├── models.py               # مدل‌های یادگیری ماشین
├── visualization.py        # نمایش نتایج و نمودارها
├── fraud_detector.py       # کلاس اصلی
├── main.py                 # اسکریپت اجرا
├── requirements.txt        # کتابخانه‌های مورد نیاز
└── README.md              # این فایل
```

## نصب

1. کلون کردن پروژه:
```bash
git clone <repository-url>
cd XGBoost Fraud Detection
```

2. نصب کتابخانه‌ها:
```bash
pip install -r requirements.txt
```

## استفاده

### اجرای ساده

```bash
python main.py
```

### اجرا با داده‌های خودتان

```bash
python main.py --data path/to/your/data.csv
```

### ذخیره نتایج و مدل‌ها

```bash
python main.py --save-results --save-models
```

### نمایش دمو پیش‌بینی

```bash
python main.py --demo
```

## استفاده در کد

```python
from fraud_detector import FraudDetector

# ایجاد detector
detector = FraudDetector()

# اجرای تحلیل کامل
results = detector.run_complete_analysis()

# پیش‌بینی برای داده جدید
predictions, probabilities = detector.predict_new_data(new_data)

# مقایسه مدل‌ها
detector.compare_models('f1_score')
```

## تنظیمات

تمام تنظیمات در فایل `config.py` قابل تغییر هستند:

- `RANDOM_STATE`: برای تکرارپذیری
- `TEST_SIZE`: نسبت داده‌های تست
- `MODEL_CONFIGS`: پارامترهای مدل‌ها
- `SAMPLE_DATA_SIZE`: اندازه داده نمونه

## معیارهای ارزیابی

- **Precision**: دقت تشخیص کلاهبرداری
- **Recall**: نرخ تشخیص موارد کلاهبرداری
- **F1-Score**: میانگین هارمونیک Precision و Recall
- **AUC**: سطح زیر منحنی ROC

## فرمت داده‌ها

داده‌های ورودی باید شامل:
- ویژگی‌های عددی (V1, V2, ..., V28)
- Time: زمان تراکنش
- Amount: مقدار تراکنش
- Class: برچسب (0: عادی، 1: کلاهبرداری)

## نتایج نمونه

```
نتایج مقایسه مدل‌ها
================================================================================
                   precision    recall  f1_score       auc  training_time
Logistic Regression    0.8756    0.8234    0.8487    0.9245         0.12
Decision Tree          0.8921    0.8456    0.8683    0.9156         0.25
Random Forest          0.9123    0.8678    0.8895    0.9387         1.45
XGBoost               0.9234    0.8789    0.9006    0.9421         2.34
Improved XGBoost      0.9345    0.8923    0.9129    0.9512         4.67

بهترین مدل بر اساس F1-Score: Improved XGBoost
F1-Score: 0.9129
```

## مراجع

پروژه بر اساس مقاله "An Improved XGBoost Model Based on Spark for Credit Card Fraud Prediction" پیاده‌سازی شده است.
