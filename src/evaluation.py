import os
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc': roc_auc_score(y, y_prob),
        'report': classification_report(y, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y, y_pred)
    }
    
    return metrics

def check_overfitting(model, X_train, y_train, X_test, y_test, name):
    
    # Train performance
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_acc = roc_auc_score(y_train, y_train_prob)

    # Test performance
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_acc = roc_auc_score(y_test, y_test_prob)

    print(f"\n{name}")
    print("Train AUC:", train_acc)
    print("Test AUC:", test_acc)
    print("Gap:", train_acc - test_acc)


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

def save_model_metrics(model_name, metrics, path):
    os.makedirs("reports", exist_ok=True)
    data = {}
    # # Load existing
    # if os.path.exists(path):
    #     with open(path, "r") as f:
    #         data = json.load(f)
    # else:
    #     data = {}

    # Convert all values

    cleaned_metrics = {k: convert_numpy(v) for k, v in metrics.items()}
    # Add current model results
    data[model_name] = cleaned_metrics

    # Save back
    with open(path, "w") as f:
        json.dump(data, f, indent=4)