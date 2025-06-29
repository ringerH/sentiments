"""
Evaluation metrics utilities
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred):
    """Compute standard classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def detailed_metrics(y_true, y_pred, labels=None):
    """Get detailed metrics including per-class breakdown"""
    metrics = compute_metrics(y_true, y_pred)
    
    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Add classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['per_class'] = report
    
    return metrics


def compare_models(results_dict):
    """Compare multiple model results"""
    comparison = {}
    
    for model_name, metrics in results_dict.items():
        comparison[model_name] = {
            'accuracy': metrics.get('accuracy', 0),
            'f1_score': metrics.get('f1_score', 0)
        }
    
    # Sort by accuracy
    sorted_models = sorted(comparison.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    return {
        'ranking': sorted_models,
        'best_model': sorted_models[0][0] if sorted_models else None
    }


def print_metrics(metrics, model_name="Model"):
    """Pretty print metrics"""
    print(f"\n{model_name} Performance:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    if 'confusion_matrix' in metrics:
        print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")


def print_comparison(comparison):
    """Pretty print model comparison"""
    print("\nModel Comparison (by accuracy):")
    print("-" * 40)
    
    for i, (model_name, scores) in enumerate(comparison['ranking'], 1):
        print(f"{i}. {model_name:15} - Acc: {scores['accuracy']:.4f}, F1: {scores['f1_score']:.4f}")
    
    print(f"\nBest Model: {comparison['best_model']}")