import os

import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from joblib import dump

load_dotenv()

# configuration
RANDOM_STATE = 42
UNCERTAIN_AS_1 = True  # how to handle UNSURE responses: True -> 1, False -> 0.5
DATA = os.environ.get("DATA")

original_df = pd.read_csv(os.path.join(DATA, "full_dataset.csv"))[["id", "MK_IN"]]
original_df = original_df.rename(columns={"id": "paper_id", "MK_IN": "label"})

llm_responses_df = pd.read_csv(os.path.join(DATA, "llm_responses.csv"))
llm_responses_df_wide = llm_responses_df.pivot(index="paper_id", columns="question_id", values="response").reset_index()

df = pd.merge(original_df, llm_responses_df_wide, on="paper_id", how="inner")

features_cols = [col for col in df.columns if col.startswith("I") or col.startswith("E")]
print(f"Number of features: {len(features_cols)}")
print(f"Features: {features_cols}")
label_col = "label"

# Helper function to convert YES/NO/UNSURE to numeric
def convert_yes_no_unsure_to_int(value: str, unsure_as_1: bool) -> int:
    """Convert Yes/No/Unsure to binary: NO->0, YES->1, UNSURE->1 or 0.5"""
    if value == "NO":
        return 0
    elif value == "YES":
        return 1
    elif value == "UNSURE":
        return 1 if unsure_as_1 else 0.5
    else:
        raise ValueError(f"Unexpected value: {value}")


df_processed = df.copy()

# Convert feature columns to numeric in a single processed DataFrame
df_processed = df.copy()
for col in features_cols:
    df_processed[col] = df_processed[col].apply(convert_yes_no_unsure_to_int, unsure_as_1=UNCERTAIN_AS_1)

# Prepare X (features) and y (labels)
X = df_processed[features_cols]
y = df_processed[label_col]

print(f"Dataset shape: {X.shape}")
print(f"\nClass distribution:")
print(y.value_counts(normalize=True).round(2))

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full)

print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

param_distributions = {
    "n_estimators": [100, 200, 350, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf": [2, 3, 4, 5],
    "bootstrap": [True, False],
}

def specificity_score(y_true, y_pred):
    """Calculate specificity (True Negative Rate)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


specificity_scorer = make_scorer(specificity_score)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "specificity": specificity_scorer,
}

N_FOLDS = 5
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=40,
    cv=cv,
    scoring=scoring,
    refit="recall",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best CV Recall Score:", round(random_search.best_score_, 3))

best_rf_rnd = random_search.best_estimator_

# Check for contradictory observations (identical features, different labels)
def resolve_conflicts(df_input, strategy="majority"):
    """
    Handle papers with identical features but different labels.

    Args:
        df_input: DataFrame with features and 'label' column
        strategy: 'remove' (drop all conflicts) or 'majority' (keep majority label)

    Returns:
        Cleaned DataFrame
    """
    feature_cols = [col for col in df_input.columns if col.startswith("I") or col.startswith("E")]

    if strategy == "remove":
        # Remove all papers with conflicting labels
        df_clean = df_input.groupby(feature_cols).filter(lambda x: x["label"].nunique() == 1)
    elif strategy == "majority":
        # Compute majority label per feature pattern avoiding groupby.apply on grouping columns
        def majority_label(series):
            counts = series.value_counts()
            if len(counts) == 1:
                return counts.index[0]
            # If tie, prefer True (to favor recall)
            if counts.iloc[0] == counts.iloc[1]:
                return True
            return counts.idxmax()

        maj = df_input.groupby(feature_cols, as_index=False)["label"].agg(majority_label).rename(columns={"label": "majority_label"})

        # Merge majority label back and keep only rows matching it
        df_merged = df_input.merge(maj, on=feature_cols, how="left")
        df_clean = df_merged[df_merged["label"] == df_merged["majority_label"]].drop(columns=["majority_label"])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df_clean.reset_index(drop=True)


# Use the processed dataset (with conflicts) for splitting
X_original = df_processed[features_cols]
y_original = df_processed["label"]

# Split ORIGINAL data first - test set will contain conflicts
X_train_orig, X_temp_orig, y_train_orig, y_temp_orig = train_test_split(
    X_original, y_original, test_size=0.4, random_state=RANDOM_STATE, stratify=y_original
)
X_val_orig, X_test_orig, y_val_orig, y_test_orig = train_test_split(
    X_temp_orig, y_temp_orig, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp_orig
)

# Combine train + val for final training
X_train_full_orig = pd.concat([X_train_orig, X_val_orig])
y_train_full_orig = pd.concat([y_train_orig, y_val_orig])

print(f"Original dataset splits (test contains conflicts):")
print(f"  Train: {len(X_train_full_orig)} papers, {y_train_full_orig.sum()} included ({y_train_full_orig.sum()/len(X_train_full_orig)*100:.1f}%)")
print(f"  Test: {len(X_test_orig)} papers, {y_test_orig.sum()} included ({y_test_orig.sum()/len(X_test_orig)*100:.1f}%)")

# Now clean ONLY the training data using majority strategy
train_df = pd.DataFrame(X_train_full_orig)
train_df["label"] = y_train_full_orig.values
train_df_clean = resolve_conflicts(train_df, strategy="majority")

X_train_clean = train_df_clean[features_cols]
y_train_clean = train_df_clean["label"]

print(f"\nAfter cleaning ONLY training data:")
print(f"  Train (clean): {len(X_train_clean)} papers, {y_train_clean.sum()} included ({y_train_clean.sum()/len(X_train_clean)*100:.1f}%)")
print(f"  Removed from train: {len(X_train_full_orig) - len(X_train_clean)} conflicting papers")

# Train with SMOTE on cleaned training data
print("\nApplying SMOTE to cleaned training data...")
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train_clean, y_train_clean)

print(f"After SMOTE:")
print(f"  Train: {len(X_train_smote)} papers, {y_train_smote.sum()} included ({y_train_smote.sum()/len(y_train_smote)*100:.1f}%)")


# Train RF on SMOTE data and save the model
rf_smote = RandomForestClassifier(**best_rf_rnd.get_params())
rf_smote.fit(X_train_smote, y_train_smote)
model_save_path = "rf_smote_model.joblib"
dump(rf_smote, model_save_path)
print(f"Saved rf_smote model to {model_save_path}")

# Predict on ORIGINAL test set (with conflicts!)
y_test_proba_smote = rf_smote.predict_proba(X_test_orig)[:, 1]

# Find best threshold for 95%+ recall
recall_results_smote = []
for thresh in np.arange(0.05, 0.55, 0.05):
    y_pred_thresh = (y_test_proba_smote >= thresh).astype(int)
    recall_results_smote.append(
        {
            "threshold": thresh,
            "recall": recall_score(y_test_orig, y_pred_thresh),
            "precision": precision_score(y_test_orig, y_pred_thresh),
            "accuracy": accuracy_score(y_test_orig, y_pred_thresh),
            "specificity": specificity_score(y_test_orig, y_pred_thresh),
        }
    )

recall_df_smote = pd.DataFrame(recall_results_smote)
best_thresh_smote = (
    recall_df_smote[recall_df_smote["recall"] >= 0.95].iloc[-1] if (recall_df_smote["recall"] >= 0.95).any() else recall_df_smote.iloc[-1]
)

print(f"\nSMOTE + Clean Training Data Results (test with conflicts):")
print(f"Best threshold: {best_thresh_smote['threshold']:.2f}")
print(f"  Recall: {best_thresh_smote['recall']:.2f}")
print(f"  Precision: {best_thresh_smote['precision']:.2f}")
print(f"  Accuracy: {best_thresh_smote['accuracy']:.2f}")
print(f"  Specificity: {best_thresh_smote['specificity']:.2f}")