# =========================================================
#   IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score
)

# =========================================================
#   LOAD DATASET
# =========================================================
print("Loading dataset...")

df = pd.read_excel(
    r"D:\Sana Rasool\Projects\AI_PBL_2025\AI_PBL2025\ASAP2_train_sourcetexts.xlsx"
)

print("Dataset Loaded Successfully!\n")
print(df.head())

# =========================================================
#   CHECK REQUIRED COLUMNS
# =========================================================
required_cols = ['full_text', 'score']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"ERROR: Column '{col}' is missing!")

## =========================================================
#   TEXT PREPROCESSING (FINAL FIX)
# =========================================================

# Keep only valid text rows
df = df.dropna(subset=['full_text', 'score'])

# Convert to string and clean
df['full_text'] = df['full_text'].astype(str).str.lower().str.strip()

# Remove empty strings
df = df[df['full_text'].str.len() > 0]

X = df['full_text']
y = df['score'].astype(int)

# =========================================================
#   LABEL ENCODING
# =========================================================
le = LabelEncoder()
y_enc = le.fit_transform(y)

# =========================================================
#   TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_enc,
    test_size=0.30,
    random_state=42,
    stratify=y_enc
)

# =========================================================
#   IMPROVED TF-IDF VECTORIZATION
# =========================================================
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=70000,
    min_df=3,
    sublinear_tf=True
)

Xtr = tfidf.fit_transform(X_train)
Xt = tfidf.transform(X_test)

# =========================================================
#   LOGISTIC REGRESSION (BEST BASELINE)
# =========================================================
lr_clf = LogisticRegression(
    max_iter=3000,
    C=2.0,
    class_weight='balanced',
    solver='lbfgs'
)

lr_clf.fit(Xtr, y_train)

y_pred_lr = lr_clf.predict(Xt)

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr, rec_lr, f1_lr, _ = precision_recall_fscore_support(
    y_test, y_pred_lr, average='macro'
)
qwk_lr = cohen_kappa_score(y_test, y_pred_lr, weights='quadratic')

print("\n===== LOGISTIC REGRESSION PERFORMANCE =====")
print("Accuracy:", round(acc_lr, 4))
print("Precision:", round(prec_lr, 4))
print("Recall:", round(rec_lr, 4))
print("F1 Score:", round(f1_lr, 4))
print("Quadratic Weighted Kappa:", round(qwk_lr, 4))

# Confusion Matrix - Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8,6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.tight_layout()
plt.show()

# =========================================================
#   TUNED SVM MODEL (AFTER IMPROVEMENT)
# =========================================================
print("\nTraining Optimized SVM model...")
start_time = time.time()

svm_clf = LinearSVC(
    C=2.0,
    class_weight='balanced',
    max_iter=5000
)

svm_clf.fit(Xtr, y_train)
svm_time = time.time() - start_time

y_pred_svm = svm_clf.predict(Xt)

acc_svm = accuracy_score(y_test, y_pred_svm)
prec_svm, rec_svm, f1_svm, _ = precision_recall_fscore_support(
    y_test, y_pred_svm, average='macro'
)
qwk_svm = cohen_kappa_score(y_test, y_pred_svm, weights='quadratic')

print("\n===== OPTIMIZED SVM PERFORMANCE =====")
print("Accuracy:", round(acc_svm, 4))
print("Precision:", round(prec_svm, 4))
print("Recall:", round(rec_svm, 4))
print("F1 Score:", round(f1_svm, 4))
print("Quadratic Weighted Kappa:", round(qwk_svm, 4))
print("Training Time (seconds):", round(svm_time, 2))

# Confusion Matrix - SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8,6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens')
plt.title('Optimized SVM Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.tight_layout()
plt.show()
