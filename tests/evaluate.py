import os

import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import httpx
from tqdm.asyncio import tqdm as async_tqdm
import asyncio
from asyncio import Semaphore
import json
from datetime import datetime

RANDOM_STATE = 42

df = pd.read_csv(os.path.join("../data","pure_dataset_splits.csv"))
test_df = df[df["split"] == "test"]

print(f"Original dataset splits (test contains conflicts):")
print(f"  Test: {len(test_df.index)} papers, {test_df['label'].sum()} included ({test_df['label'].sum()/len(test_df.index)*100:.1f}%)")

def specificity_score(y_true, y_pred):
    """Calculate specificity (True Negative Rate)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Predict on ORIGINAL test set (with conflicts!)

async def fetch_prediction(client, url, label, item, sem):
    payload = {'title': '', 'abstract': ''}
    if item['title'] is not None and not pd.isna(item['title']):
        payload["title"] = item['title']
    if item['abstract'] is not None and not pd.isna(item['abstract']):
        payload["abstract"] = item['abstract']
    
    async with sem:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        # Include the ground truth label in the stored data
        data['ground_truth_label'] = label
        if label == 1 and data['prediction'] == 0:
            data['prediction_class'] = "FN"
        elif label == 1 and data['prediction'] == 1:
            data['prediction_class'] = "TP"
        elif label == 0 and data['prediction'] == 1:
            data['prediction_class'] = "FP"
        elif label == 0 and data['prediction'] == 0:
            data['prediction_class'] = "TN"
        return data

async def evaluate():
    url = f"http://localhost:8000/prediction"
    all_full_responses = [] # List to store full JSON objects
    sem = Semaphore(8) 
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        tasks = []

        counter = 0
        
        for index, row in test_df.iterrows():
            tasks.append(fetch_prediction(client, url, int(row['label']), row, sem))
            if counter >= 100:
                break
            counter += 1
        
        for result in async_tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            full_data = await result
            all_full_responses.append(full_data)

    # Save to JSON file
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"responses_{timestamp}.json"
    output_path = os.path.join("../logs", filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_full_responses, f, indent=4)
    
    print(f"Saved {len(all_full_responses)} responses to {output_path}")

    for response in all_full_responses:
        if response['ground_truth_label'] == 1 and response['prediction'] == 0:
            print(response)

    # --- Metrics Calculation ---
    y_test_gt = [r['ground_truth_label'] for r in all_full_responses]
    y_test_proba_smote = [r['probability'] for r in all_full_responses]
    y_test_pred_smote = [r['prediction'] for r in all_full_responses]

    # Find best threshold for 95%+ recall
    recall_results_smote = []
    for thresh in np.arange(0.05, 0.55, 0.05):
        y_pred_thresh = (y_test_proba_smote >= thresh).astype(int)
        recall_results_smote.append(
            {
                "threshold": thresh,
                "recall": recall_score(y_test_gt, y_pred_thresh),
                "precision": precision_score(y_test_gt, y_pred_thresh),
                "accuracy": accuracy_score(y_test_gt, y_pred_thresh),
                "specificity": specificity_score(y_test_gt, y_pred_thresh),
            }
        )

    recall_df_smote = pd.DataFrame(recall_results_smote)
    best_thresh_smote = (
        recall_df_smote[recall_df_smote["recall"] >= 0.95].iloc[-1]
        if (recall_df_smote["recall"] >= 0.95).any()
        else recall_df_smote.loc[recall_df_smote["recall"].idxmax()]
    )

    print("Endpoint prediction recall: ", recall_score(y_test_gt, y_test_pred_smote))
    print("Endpoint prediction precision: ", precision_score(y_test_gt, y_test_pred_smote))
    print("Endpoint prediction specificity: ", specificity_score(y_test_gt, y_test_pred_smote))

    print(f"Best threshold: {best_thresh_smote['threshold']:.2f}")
    print(f"  Recall: {best_thresh_smote['recall']:.2f}")
    print(f"  Precision: {best_thresh_smote['precision']:.2f}")
    print(f"  Specificity: {best_thresh_smote['specificity']:.2f}")

asyncio.run(evaluate())