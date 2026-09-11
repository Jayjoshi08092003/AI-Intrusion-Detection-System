import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import shapiro, ttest_rel, wilcoxon

# ==========================================
# 1. LOAD LEAK-FREE DATA
# ==========================================
# We run Cross-Validation ONLY on the training data to keep the 10% holdout sacred.
try:
    X_train = np.load("processed_data/l4_X_train.npy")
    y_train = np.load("processed_data/l4_y_train.npy")
except FileNotFoundError:
    print("[FATAL] Data not found. Run the preprocessing pipeline first.")
    exit()

print(f"[*] Loaded Training Data for CV: {X_train.shape[0]} samples")

# ==========================================
# 2. DEFINE MODELS
# ==========================================
# Your proposed architecture
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42
)

# Standard baseline for comparison
rf_baseline = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

# ==========================================
# 3. EXECUTE K-FOLD CROSS VALIDATION
# ==========================================
K = 5
print(f"\n[*] Executing {K}-Fold Cross-Validation...")
cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

# Collect F1 scores for both models across all 5 folds
xgb_f1_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
rf_f1_scores = cross_val_score(rf_baseline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

print("\n--- Cross-Validation F1 Scores ---")
print(f"XGBoost (Proposed) : {xgb_f1_scores}")
print(f"Random Forest (Base): {rf_f1_scores}")
print(f"XGB Mean F1: {np.mean(xgb_f1_scores):.5f} | RF Mean F1: {np.mean(rf_f1_scores):.5f}")

# ==========================================
# 4. STATISTICAL HYPOTHESIS TESTING
# ==========================================
print("\n[*] Performing Statistical Significance Testing...")

# Calculate the difference between scores
score_differences = xgb_f1_scores - rf_f1_scores

# Test for Normality using Shapiro-Wilk Test
# H0: The differences are normally distributed
stat, p_shapiro = shapiro(score_differences)
print(f"Shapiro-Wilk Test p-value: {p_shapiro:.4f}")

alpha = 0.05
if p_shapiro > alpha:
    print("[*] Differences are normally distributed. Using Paired t-Test.")
    # H0: The mean difference between the two models is zero.
    t_stat, p_value = ttest_rel(xgb_f1_scores, rf_f1_scores)
    test_used = "Paired t-Test"
else:
    print("[*] Differences are NOT normally distributed. Using Wilcoxon Signed-Rank Test.")
    # H0: The median difference between the two models is zero.
    w_stat, p_value = wilcoxon(xgb_f1_scores, rf_f1_scores)
    test_used = "Wilcoxon Signed-Rank Test"
    t_stat = w_stat # Storing for output consistency

# ==========================================
# 5. Q1 JOURNAL OUTPUT FORMATTING
# ==========================================
print("\n" + "=" * 50)
print(" STATISTICAL VALIDATION RESULTS FOR SECTION 6 ")
print("=" * 50)
print(f"Metric Evaluated : F1-Score")
print(f"Test Applied     : {test_used}")
print(f"Test Statistic   : {t_stat:.4f}")
print(f"p-value          : {p_value:.6f}")

if p_value < 0.01:
    print("Conclusion       : HIGHLY SIGNIFICANT (p < 0.01)")
    print("Interpretation   : Your model mathematically dominates the baseline. Write this in the paper.")
elif p_value < 0.05:
    print("Conclusion       : SIGNIFICANT (p < 0.05)")
    print("Interpretation   : Your model is statistically better. Safe to publish.")
else:
    print("Conclusion       : NOT SIGNIFICANT (p >= 0.05)")
    print("Interpretation   : Brutal truth - your model's edge is a statistical illusion. Back to the drawing board.")
print("=" * 50)