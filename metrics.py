import numpy as np
import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_metric(predictions):
    # Handle empty predictions list
    if not predictions:
        return 0.0
    if isinstance(predictions[0].correct_candidate, list):
        return np.mean([pred.predicted_candidate in pred.correct_candidate for pred in predictions])
    else:
        return np.mean([pred.correct_candidate == pred.predicted_candidate for pred in predictions])


def print_confusion_matrix(predictions, eval_samples=None):
    """
    Print confusion matrix to verify model predictions.
    
    Works for all classification datasets (SST2, CB, BoolQ, RTE, WIC, WSC, etc.)
    
    Args:
        predictions: List of Prediction objects with correct_candidate and predicted_candidate
        eval_samples: Optional list of eval samples to get candidate names for labels
    
    Returns:
        dict with confusion matrix metrics (macro_f1, per_class_f1, etc.)
    """
    # Extract ground truth and predicted labels
    y_true = []
    y_pred = []
    
    for pred in predictions:
        # Handle both single and list correct_candidate
        if isinstance(pred.correct_candidate, list):
            y_true.append(pred.correct_candidate[0])
        else:
            y_true.append(pred.correct_candidate)
        y_pred.append(pred.predicted_candidate)
    
    # Get unique labels
    all_labels = sorted(set(y_true) | set(y_pred))
    n_labels = len(all_labels)
    
    # Get candidate names from first sample if available
    label_names = None
    if eval_samples and hasattr(eval_samples[0], 'candidates'):
        candidates = eval_samples[0].candidates
        if len(candidates) >= n_labels:
            label_names = [str(candidates[i])[:15] for i in all_labels]
    
    if label_names is None:
        label_names = [str(i) for i in all_labels]
    
    # Build confusion matrix
    confusion = {}
    for t, p in zip(y_true, y_pred):
        if t not in confusion:
            confusion[t] = Counter()
        confusion[t][p] += 1
    
    # Calculate column width based on label names
    col_width = max(12, max(len(name) for name in label_names) + 2)
    header_width = max(15, max(len(name) for name in label_names) + 2)
    total_width = header_width + 3 + (col_width + 3) * n_labels + 8
    
    # Print header
    print("\n" + "=" * total_width)
    print("CONFUSION MATRIX")
    print("=" * total_width)
    header_label = "Actual \\ Pred"
    print(f"{header_label:>{header_width}} | " + " | ".join(f"{name:>{col_width}}" for name in label_names) + " | Total")
    print("-" * total_width)
    
    # Print rows
    for i, true_label in enumerate(all_labels):
        row_counts = [confusion.get(true_label, {}).get(pred_label, 0) for pred_label in all_labels]
        row_total = sum(row_counts)
        print(f"{label_names[i]:>{header_width}} | " + " | ".join(f"{c:>{col_width}}" for c in row_counts) + f" | {row_total:>5}")
    
    print("-" * total_width)
    
    # Print column totals
    col_totals = []
    for pred_label in all_labels:
        total = sum(confusion.get(t, {}).get(pred_label, 0) for t in all_labels)
        col_totals.append(total)
    print(f"{'Total':>{header_width}} | " + " | ".join(f"{c:>{col_width}}" for c in col_totals) + f" | {sum(col_totals):>5}")
    
    # Print per-class metrics
    print("\n" + "-" * total_width)
    print("PER-CLASS METRICS:")
    print("-" * total_width)
    
    f1_scores = []
    per_class_metrics = {}
    
    for i, label in enumerate(all_labels):
        tp = confusion.get(label, {}).get(label, 0)
        fn = sum(confusion.get(label, {}).values()) - tp
        fp = sum(confusion.get(t, {}).get(label, 0) for t in all_labels) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
        per_class_metrics[label_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }
        
        print(f"  {label_names[i]:>{header_width}}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f} (support={tp + fn})")
    
    # Compute macro F1
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    # Compute weighted F1
    supports = [confusion.get(label, {}).get(label, 0) + sum(confusion.get(label, {}).values()) - confusion.get(label, {}).get(label, 0) 
                for label in all_labels]
    total_support = sum(supports)
    weighted_f1 = sum(f1 * sup for f1, sup in zip(f1_scores, supports)) / total_support if total_support > 0 else 0
    
    print("-" * total_width)
    print(f"MACRO F1 SCORE:    {macro_f1:.4f}")
    print(f"WEIGHTED F1 SCORE: {weighted_f1:.4f}")
    
    # Print prediction distribution summary
    pred_dist = Counter(y_pred)
    true_dist = Counter(y_true)
    print("\n" + "-" * total_width)
    print("DISTRIBUTION SUMMARY:")
    print(f"  Ground Truth: {dict(sorted(true_dist.items()))}")
    print(f"  Predictions:  {dict(sorted(pred_dist.items()))}")
    
    # Warning if model predicts only one class
    if len(pred_dist) == 1:
        majority_class = list(pred_dist.keys())[0]
        print(f"\n  ⚠️  WARNING: Model predicts ONLY class {majority_class} ({label_names[all_labels.index(majority_class)]})")
        print(f"      This suggests the model is NOT learning and just predicting the majority class!")
    
    print("=" * total_width + "\n")
    
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_metrics': per_class_metrics,
        'confusion': {str(k): dict(v) for k, v in confusion.items()}
    }