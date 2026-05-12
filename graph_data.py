# Graph data
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def graph_data(data_frame, features):
    fig, axes = plt.subplots(2, 3, figsize = (15, 10))
    
    for idx, feature in enumerate(features):
        ax = axes[idx // 3, idx % 3]
        normal = data_frame[data_frame['is-anomaly'] == False][feature]
        anomaly = data_frame[data_frame['is-anomaly'] == True][feature]

        ax.hist(normal, alpha = 0.6, label = 'Normal', bins = 30)
        ax.hist(anomaly, alpha = 0.6, label = 'Anomaly', bins = 30)
        ax.set_title(feature)
        ax.legend()
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi = 300, bbox_inches = 'tight')
    # plt.show()

def graph_scores(models_dict):
    """
    Graph comparison of model metrics (Accuracy, Sensitivity, Specificity, F1-Score, Recall)
    
    models_dict: Dictionary with structure:
    {
        'Model Name': {'accuracy': value, 'sensitivity': value, 'specificity': value, 'f1': value, 'recall': value},
        ...
    }
    """
    import numpy as np
    
    model_names = list(models_dict.keys())
    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'recall']
    
    # Extract values for each metric
    x = np.arange(len(model_names))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create bars for each metric
    for idx, metric in enumerate(metrics):
        values = [models_dict[model][metric] for model in model_names]
        ax.bar(x + idx * width, values, width, label=metric.upper())
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi = 300, bbox_inches = 'tight')
    # plt.show()
