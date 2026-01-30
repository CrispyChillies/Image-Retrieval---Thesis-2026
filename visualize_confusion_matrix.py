import numpy as np
import torch
from plot_confusion_matrix import plot_confusion_matrix
from collections import Counter

def get_majority_vote_predictions(labels, dists, k=1):
    """
    Returns true and predicted labels using majority vote for top-k retrieval.
    Args:
        labels (Tensor): ground truth labels
        dists (Tensor): distance matrix (higher = more similar)
        k (int): top-k retrieved images
    Returns:
        true_labels (list), predicted_labels (list)
    """
    labels_np = labels.cpu().numpy()
    ranks = torch.argsort(dists, dim=0, descending=True).cpu().numpy()
    predicted_labels = []
    true_labels = []
    for i in range(labels_np.shape[0]):
        top_k_indices = ranks[:k, i]
        retrieved_labels = labels_np[top_k_indices]
        # Majority vote
        counter = Counter(retrieved_labels)
        pred_label = counter.most_common(1)[0][0]
        predicted_labels.append(pred_label)
        true_labels.append(labels_np[i])
    return true_labels, predicted_labels

if __name__ == "__main__":
    # Example usage: load npz file and plot confusion matrix
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, required=True, help='Path to .npz file with labels and dists')
    parser.add_argument('--k', type=int, default=1, help='Top-k for majority vote')
    parser.add_argument('--save', type=str, default=None, help='Path to save confusion matrix image')
    parser.add_argument('--show', action='store_true', help='Show the plot')
    args = parser.parse_args()

    data = np.load(args.npz)
    labels = torch.tensor(data['labels'])
    dists = -torch.tensor(data['dists'])  # Negate if stored as -dists
    class_names = ['No Finding', 'Pneumonia', 'COVID-19']
    true_labels, predicted_labels = get_majority_vote_predictions(labels, dists, k=args.k)
    plot_confusion_matrix(true_labels, predicted_labels, class_names, save_path=args.save, show=args.show,
                         title=f'Confusion Matrix (k={args.k})')
