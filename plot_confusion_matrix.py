import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(true_labels, predicted_labels, class_names, save_path=None, show=True, title=None):
    """
    Plots and optionally saves a confusion matrix.
    Args:
        true_labels (array-like): Ground truth labels
        predicted_labels (array-like): Predicted labels
        class_names (list): List of class names (for axis labels)
        save_path (str, optional): If provided, saves the plot to this path
        show (bool): Whether to display the plot
        title (str, optional): Title for the plot
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
