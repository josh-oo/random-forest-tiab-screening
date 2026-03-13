import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTEN
from imblearn.pipeline import Pipeline as ImbPipeline  # Use imblearn's pipeline
from joblib import dump

load_dotenv()

# Configuration
RANDOM_STATE = 42
DATA = os.environ.get("DATA")

df = pd.read_csv(os.path.join(DATA, "pure_dataset_splits.csv"))

#df = df.drop(["I6", "I2", "I9", "I3", "I14", "I8", "I11", "I12", "I15"], axis=1)
#df = df.drop(["E10", "E4", "E11", "E8", "E7", "E5", "E6", "E1", "E12", "E9", "E3", "E2"], axis=1)
#df = df.drop(["I15"], axis=1)
#df = df.drop(["I13", "I3", "I1", "I4", "I5", "I10", "I16", "I9", "I14", "I7", "I17", "I6", "I2", "I15", "I8", "I12","E4","I11"], axis=1)

def convert_yes_no_unsure_to_int(value: str, is_include: bool) -> int:
    if value == "NO": return 0
    elif value == "YES": return 1
    elif value == "UNSURE": return 1 if is_include else 0
    else: raise ValueError(f"Unexpected value: {value}")

features_cols = [col for col in df.columns if col.startswith("I") or col.startswith("E")]
features_cols = ['I1', 'I10', 'I13', 'I16', 'I17', 'I4', 'I5', 'I7']
label_col = "label"

for col in features_cols:
    df[col] = df[col].apply(convert_yes_no_unsure_to_int, is_include=col.upper().startswith("I"))

# 4. Filter by the "split" column
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

# 5. Extract X and y 
X_train_clean = train_df[features_cols]
y_train_clean = train_df["label"]

X_test_orig = test_df[features_cols]
y_test_orig = test_df["label"]

# 3. Create Imblearn Pipeline
pipeline = ImbPipeline([
    ('smote', SMOTEN(random_state=RANDOM_STATE, k_neighbors=5)),
    ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE))
])

# 4. Define parameter grid (Note the 'classifier__' prefix and removal of 'class_weight')
param_distributions = {
    "classifier__max_depth": [None],#, 10, 20, 21, 22, 23, 24, 25, 30, 50, 75, 100, 150],
    "classifier__min_samples_split": [2, 3, 4, 5, 10, 20, 30, 35, 40, 45, 50, 60, 70],
    "classifier__min_samples_leaf": [1, 2, 3, 5, 10, 20, 50],
    "classifier__criterion": ["gini", "entropy", "log_loss"],
    "classifier__max_features": [None, "sqrt", "log2"],
}

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

specificity_scorer = make_scorer(specificity_score)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "specificity": specificity_scorer,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 5. Run Search
random_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_distributions,
    cv=cv,
    scoring=scoring,
    refit="recall",
    #refit="precision",
    n_jobs=-1,
    verbose=2
)

print("\nStarting Hyperparameter Search with SMOTE Pipeline...")
random_search.fit(X_train_clean, y_train_clean)

print("Best Parameters:", random_search.best_params_)
print("Best CV Recall Score:", round(random_search.best_score_, 5))

# 6. Save the pipeline model
_, best_model = random_search.best_estimator_.steps[1]
model_save_path = "dt_smote_pipeline_model.joblib"
dump(best_model, model_save_path)
print(f"Saved pipeline model to {model_save_path}")

# 7. Predict on ORIGINAL test set (Pipeline handles bypassing SMOTE during prediction)
y_test_proba_smote = best_model.predict_proba(X_test_orig)[:, 1]

# Find best threshold for 95%+ recall
recall_results_smote = []
for thresh in np.arange(0.05, 0.55, 0.05):
    y_pred_thresh = (y_test_proba_smote >= thresh).astype(int)
    recall_results_smote.append({
        "threshold": thresh,
        "recall": recall_score(y_test_orig, y_pred_thresh),
        "precision": precision_score(y_test_orig, y_pred_thresh),
        "accuracy": accuracy_score(y_test_orig, y_pred_thresh),
        "specificity": specificity_score(y_test_orig, y_pred_thresh),
    })

recall_df_smote = pd.DataFrame(recall_results_smote)
best_thresh_smote = (
    recall_df_smote[recall_df_smote["recall"] >= 0.95].iloc[-1]
    if (recall_df_smote["recall"] >= 0.95).any()
    else recall_df_smote.loc[recall_df_smote["recall"].idxmax()]
)

print(features_cols)

print(f"\nPipeline Results (Test with conflicts):")
print(f"Best threshold: {best_thresh_smote['threshold']:.2f}")
print(f"  Recall: {best_thresh_smote['recall']:.2f}")
print(f"  Precision: {best_thresh_smote['precision']:.2f}")

# 8. Evaluation Section
# Extract the actual decision tree from the pipeline to access tree attributes
dt_classifier = best_model

leaf_indices = dt_classifier.apply(X_test_orig)
leaf_ginis = dt_classifier.tree_.impurity[leaf_indices]
best_threshold = 0.5 

def evaluate_subset(mask, label, print_it=False):
    if not np.any(mask):
        print(f"\n--- {label} --- \nNo samples in this category.")
        return
    
    y_true_sub = y_test_orig[mask]
    y_prob_sub = y_test_proba_smote[mask]
    y_pred_sub = (y_prob_sub >= best_threshold).astype(int)
    
    print(f"\n--- {label} (N={len(y_true_sub)}) ---")
    #print(f"Accuracy:    {accuracy_score(y_true_sub, y_pred_sub):.4f}")
    print(f"Recall:      {recall_score(y_true_sub, y_pred_sub, zero_division=0):.4f}")
    print(f"Precision:   {precision_score(y_true_sub, y_pred_sub, zero_division=0):.4f}")
    try:
        print(f"Specificity: {specificity_score(y_true_sub, y_pred_sub):.4f}")
    except:
        print("Specificity: N/A (Single class present)")

mask_gini_zero = (leaf_ginis == 0.0)
mask_gini_not_zero = (leaf_ginis > 0.0)
mask_all = np.ones(len(y_test_orig), dtype=bool)

evaluate_subset(mask_gini_zero, "1. ONLY GINI 0.0 LEAF NODES (Pure)", print_it=True)
evaluate_subset(mask_gini_not_zero, "2. ONLY GINI != 0.0 LEAF NODES (Impure)")
evaluate_subset(mask_all, "3. EVERYTHING TOGETHER")