"""
Backbone Comparison and Debugging Pipeline for Medical Image Retrieval

Purpose:
- Compare ConvNeXtV2 vs ConceptCLIP retrieval behavior
- Analyze retrieval overlap and rank correlation
- Test late fusion strategies
- Visualize retrieval differences

Usage:
    python test_backbone.py \
        --convnext-embeddings path/to/convnext_embeddings.npy \
        --conceptclip-embeddings path/to/conceptclip_embeddings.npy \
        --labels path/to/labels.npy \
        --image-ids path/to/image_ids.npy \
        --image-dir path/to/images \
        --save-dir results/backbone_comparison
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from collections import Counter
from PIL import Image
import torch


def load_embeddings(model1_path, model2_path, labels_path, model1_name="Model1", model2_name="Model2", image_ids_path=None):
    """
    Load pre-extracted embeddings and labels.
    
    Args:
        model1_path: Path to first model embeddings (.npy)
        model2_path: Path to second model embeddings (.npy)
        labels_path: Path to labels (.npy)
        model1_name: Name of first model for display
        model2_name: Name of second model for display
        image_ids_path: Optional path to image IDs (.npy)
    
    Returns:
        model1_emb: [N, D1] normalized embeddings
        model2_emb: [N, D2] normalized embeddings
        labels: [N] class labels
        image_ids: [N] image identifiers (or indices if not provided)
    """
    print("Loading embeddings...")
    
    model1_emb = np.load(model1_path)
    model2_emb = np.load(model2_path)
    labels = np.load(labels_path)
    
    if image_ids_path and os.path.exists(image_ids_path):
        image_ids = np.load(image_ids_path)
    else:
        image_ids = np.arange(len(labels))
    
    # Ensure embeddings are L2 normalized
    model1_emb = model1_emb / (np.linalg.norm(model1_emb, axis=1, keepdims=True) + 1e-8)
    model2_emb = model2_emb / (np.linalg.norm(model2_emb, axis=1, keepdims=True) + 1e-8)
    
    print(f"{model1_name} embeddings: {model1_emb.shape}")
    print(f"{model2_name} embeddings: {model2_emb.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Unique classes: {np.unique(labels)}")
    
    return model1_emb, model2_emb, labels, image_ids


def compute_similarity_matrix(query_emb, db_emb):
    """
    Compute cosine similarity matrix between queries and database.
    
    Args:
        query_emb: [N_q, D] normalized embeddings
        db_emb: [N_db, D] normalized embeddings
    
    Returns:
        sim_matrix: [N_q, N_db] cosine similarity scores
    """
    # For normalized embeddings, cosine similarity = dot product
    return np.dot(query_emb, db_emb.T)


def retrieve_topk(sim_matrix, k=5, exclude_self=True):
    """
    Retrieve top-k most similar images for each query.
    
    Args:
        sim_matrix: [N_q, N_db] similarity scores
        k: Number of retrievals
        exclude_self: Whether to exclude self-matches (query == db)
    
    Returns:
        topk_indices: [N_q, k] indices of top-k retrievals
        topk_scores: [N_q, k] similarity scores of top-k retrievals
    """
    if exclude_self:
        # Set diagonal to -inf to exclude self-matches
        sim_matrix = sim_matrix.copy()
        np.fill_diagonal(sim_matrix, -np.inf)
    
    # Get top-k indices (descending order)
    topk_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
    
    # Get corresponding scores
    topk_scores = np.take_along_axis(sim_matrix, topk_indices, axis=1)
    
    return topk_indices, topk_scores


def compute_ap(retrieved_labels, query_label, k=5):
    """
    Compute Average Precision for a single query.
    
    Args:
        retrieved_labels: [k] labels of retrieved images
        query_label: scalar label of query image
        k: number of retrievals
    
    Returns:
        ap: Average Precision
    """
    # Find relevant positions (1-based)
    relevant = (retrieved_labels == query_label).astype(float)
    
    if relevant.sum() == 0:
        return 0.0
    
    # Compute precision at each position
    precisions = np.cumsum(relevant) / np.arange(1, k + 1)
    
    # Average precision = mean of precisions at relevant positions
    ap = np.sum(precisions * relevant) / relevant.sum()
    
    return ap


def compute_map_at_k(topk_indices, query_labels, db_labels, k=5):
    """
    Compute Mean Average Precision @ k.
    
    Args:
        topk_indices: [N_q, k] indices of top-k retrievals
        query_labels: [N_q] query labels
        db_labels: [N_db] database labels
        k: number of retrievals
    
    Returns:
        mAP: Mean Average Precision
        aps: [N_q] individual AP scores
    """
    n_queries = len(query_labels)
    aps = np.zeros(n_queries)
    
    for i in range(n_queries):
        retrieved_labels = db_labels[topk_indices[i, :k]]
        aps[i] = compute_ap(retrieved_labels, query_labels[i], k)
    
    mAP = np.mean(aps)
    return mAP, aps


def compute_retrieval_overlap(topk_indices1, topk_indices2, k=5):
    """
    Compute retrieval overlap between two models.
    
    Args:
        topk_indices1: [N_q, k] top-k indices from model 1
        topk_indices2: [N_q, k] top-k indices from model 2
        k: number of retrievals
    
    Returns:
        overlaps: [N_q] overlap counts (0 to k)
        mean_overlap: mean overlap across queries
    """
    n_queries = topk_indices1.shape[0]
    overlaps = np.zeros(n_queries)
    
    for i in range(n_queries):
        set1 = set(topk_indices1[i, :k])
        set2 = set(topk_indices2[i, :k])
        overlaps[i] = len(set1.intersection(set2))
    
    mean_overlap = np.mean(overlaps)
    return overlaps, mean_overlap


def compute_rank_correlation(sim_matrix1, sim_matrix2):
    """
    Compute Spearman rank correlation between two similarity rankings.
    
    Args:
        sim_matrix1: [N_q, N_db] similarity scores from model 1
        sim_matrix2: [N_q, N_db] similarity scores from model 2
    
    Returns:
        correlations: [N_q] Spearman rho for each query
        mean_rho: mean correlation
        std_rho: std of correlation
    """
    n_queries = sim_matrix1.shape[0]
    correlations = np.zeros(n_queries)
    
    for i in range(n_queries):
        # Compute Spearman correlation for this query's rankings
        rho, _ = spearmanr(sim_matrix1[i], sim_matrix2[i])
        correlations[i] = rho if not np.isnan(rho) else 0.0
    
    mean_rho = np.mean(correlations)
    std_rho = np.std(correlations)
    
    return correlations, mean_rho, std_rho


def plot_overlap_histogram(overlaps, model1_name, model2_name, save_path=None):
    """
    Plot histogram of retrieval overlaps.
    
    Args:
        overlaps: [N_q] overlap counts
        model1_name: Name of first model
        model2_name: Name of second model
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    bins = np.arange(-0.5, 6.5, 1)  # 0, 1, 2, 3, 4, 5
    plt.hist(overlaps, bins=bins, edgecolor='black', alpha=0.7)
    
    plt.xlabel('Overlap Count (out of 5)', fontsize=12)
    plt.ylabel('Number of Queries', fontsize=12)
    plt.title(f'Retrieval Overlap Distribution\n({model1_name} vs {model2_name})', fontsize=14)
    plt.xticks(range(6))
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics
    mean_overlap = np.mean(overlaps)
    pct_low_overlap = np.sum(overlaps <= 2) / len(overlaps) * 100
    
    textstr = f'Mean: {mean_overlap:.2f}\nOverlap ≤ 2: {pct_low_overlap:.1f}%'
    plt.text(0.98, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved overlap histogram to {save_path}")
    
    plt.close()


def visualize_retrieval_comparison(query_idx, 
                                   model1_indices, model1_scores,
                                   model2_indices, model2_scores,
                                   labels, image_ids, image_dir,
                                   model1_name="Model1", model2_name="Model2",
                                   class_names=None, k=5, save_path=None):
    """
    Visualize retrieval comparison for a single query.
    
    Args:
        query_idx: Index of query image
        model1_indices: [k] top-k indices from first model
        model1_scores: [k] similarity scores from first model
        model2_indices: [k] top-k indices from second model
        model2_scores: [k] similarity scores from second model
        labels: [N] class labels
        image_ids: [N] image identifiers
        image_dir: Directory containing images
        model1_name: Name of first model
        model2_name: Name of second model
        class_names: Dict mapping label to class name
        k: Number of retrievals to show
        save_path: Path to save figure
    """
    if class_names is None:
        class_names = {0: 'COVID', 1: 'Pneumonia', 2: 'Normal'}
    
    fig, axes = plt.subplots(3, k + 1, figsize=(3 * (k + 1), 9))
    
    # Row 0: Query image
    query_label = labels[query_idx]
    query_id = image_ids[query_idx]
    
    # Load and display query image
    query_img_path = os.path.join(image_dir, f"{query_id}.png")
    if not os.path.exists(query_img_path):
        query_img_path = os.path.join(image_dir, f"{query_id}.jpg")
    
    if os.path.exists(query_img_path):
        query_img = Image.open(query_img_path).convert('RGB')
        axes[0, 0].imshow(query_img)
    else:
        axes[0, 0].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
    
    axes[0, 0].set_title(f'Query\n{class_names.get(query_label, query_label)}\nID: {query_id}', 
                        fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Hide remaining cells in query row
    for j in range(1, k + 1):
        axes[0, j].axis('off')
    
    # Row 1: First model retrievals
    axes[1, 0].text(0.5, 0.5, f'{model1_name}\nTop-5', ha='center', va='center',
                   fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    for j in range(k):
        idx = model1_indices[j]
        score = model1_scores[j]
        label = labels[idx]
        img_id = image_ids[idx]
        
        # Load image
        img_path = os.path.join(image_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{img_id}.jpg")
        
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            axes[1, j + 1].imshow(img)
        else:
            axes[1, j + 1].text(0.5, 0.5, 'Not Found', ha='center', va='center')
        
        # Color code: green if correct class, red otherwise
        color = 'green' if label == query_label else 'red'
        axes[1, j + 1].set_title(f'{class_names.get(label, label)}\nSim: {score:.3f}',
                                fontsize=9, color=color)
        axes[1, j + 1].axis('off')
    
    # Row 2: Second model retrievals
    axes[2, 0].text(0.5, 0.5, f'{model2_name}\nTop-5', ha='center', va='center',
                   fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    
    for j in range(k):
        idx = model2_indices[j]
        score = model2_scores[j]
        label = labels[idx]
        img_id = image_ids[idx]
        
        # Load image
        img_path = os.path.join(image_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{img_id}.jpg")
        
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            axes[2, j + 1].imshow(img)
        else:
            axes[2, j + 1].text(0.5, 0.5, 'Not Found', ha='center', va='center')
        
        # Color code: green if correct class, red otherwise
        color = 'green' if label == query_label else 'red'
        axes[2, j + 1].set_title(f'{class_names.get(label, label)}\nSim: {score:.3f}',
                                fontsize=9, color=color)
        axes[2, j + 1].axis('off')
    
    plt.suptitle(f'Retrieval Comparison - Query {query_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()


def late_fusion_retrieval(sim_matrix1, sim_matrix2, alpha=0.5):
    """
    Perform late fusion of similarity scores.
    
    Args:
        sim_matrix1: [N_q, N_db] similarity from model 1
        sim_matrix2: [N_q, N_db] similarity from model 2
        alpha: Fusion weight (alpha * model1 + (1-alpha) * model2)
    
    Returns:
        fused_sim_matrix: [N_q, N_db] fused similarity scores
    """
    return alpha * sim_matrix1 + (1 - alpha) * sim_matrix2


def print_summary_table(results, model1_name, model2_name):
    """
    Print a formatted summary table of results.
    
    Args:
        results: Dict containing evaluation metrics
        model1_name: Name of first model
        model2_name: Name of second model
    """
    print("\n" + "="*70)
    print("BACKBONE COMPARISON SUMMARY")
    print("="*70)
    
    # Header
    print(f"{'Model':<25} {'mAP@5':<12} {'Mean Overlap@5':<18} {'Spearman ρ':<15}")
    print("-"*70)
    
    # Model 1
    print(f"{model1_name:<25} {results['model1_map']:<12.4f} {'-':<18} {'-':<15}")
    
    # Model 2
    print(f"{model2_name:<25} {results['model2_map']:<12.4f} "
          f"{results['mean_overlap']:<18.4f} {results['mean_spearman']:<15.4f}")
    
    # Fusion results
    if 'fusion_results' in results:
        print("-"*70)
        for alpha, mAP in results['fusion_results'].items():
            model_name = f"Fusion (α={alpha:.1f})"
            print(f"{model_name:<25} {mAP:<12.4f} {'-':<18} {'-':<15}")
    
    print("="*70)
    
    # Additional statistics
    print("\nAdditional Statistics:")
    print(f"  Mean Spearman ρ: {results['mean_spearman']:.4f} ± {results['std_spearman']:.4f}")
    print(f"  Overlap ≤ 2 queries: {results['pct_low_overlap']:.2f}%")
    print(f"  Total queries: {results['n_queries']}")
    print("="*70 + "\n")


def main(args):
    """Main evaluation pipeline."""
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Load embeddings
    print("\n" + "="*70)
    print("STEP 1: LOADING EMBEDDINGS")
    print("="*70)
    
    model1_emb, model2_emb, labels, image_ids = load_embeddings(
        args.model1_embeddings,
        args.model2_embeddings,
        args.labels,
        args.model1_name,
        args.model2_name,
        args.image_ids
    )
    
    n_samples = len(labels)
    
    # 2. Compute similarity matrices
    print("\n" + "="*70)
    print("STEP 2: COMPUTING SIMILARITY MATRICES")
    print("="*70)
    
    print(f"Computing {args.model1_name} similarities...")
    sim_model1 = compute_similarity_matrix(model1_emb, model1_emb)
    
    print(f"Computing {args.model2_name} similarities...")
    sim_model2 = compute_similarity_matrix(model2_emb, model2_emb)
    
    # 3. Retrieve top-k
    print("\n" + "="*70)
    print(f"STEP 3: RETRIEVING TOP-{args.k}")
    print("="*70)
    
    topk_model1, scores_model1 = retrieve_topk(sim_model1, k=args.k, exclude_self=True)
    topk_model2, scores_model2 = retrieve_topk(sim_model2, k=args.k, exclude_self=True)
    
    print(f"{args.model1_name} top-{args.k} shape: {topk_model1.shape}")
    print(f"{args.model2_name} top-{args.k} shape: {topk_model2.shape}")
    
    # 4. Compute mAP@k
    print("\n" + "="*70)
    print(f"STEP 4: COMPUTING mAP@{args.k}")
    print("="*70)
    
    map_model1, aps_model1 = compute_map_at_k(topk_model1, labels, labels, k=args.k)
    map_model2, aps_model2 = compute_map_at_k(topk_model2, labels, labels, k=args.k)
    
    print(f"{args.model1_name} mAP@{args.k}: {map_model1:.4f}")
    print(f"{args.model2_name} mAP@{args.k}: {map_model2:.4f}")
    
    # 5. Compute retrieval overlap
    print("\n" + "="*70)
    print(f"STEP 5: COMPUTING RETRIEVAL OVERLAP@{args.k}")
    print("="*70)
    
    overlaps, mean_overlap = compute_retrieval_overlap(topk_model1, topk_model2, k=args.k)
    pct_low_overlap = np.sum(overlaps <= 2) / len(overlaps) * 100
    
    print(f"Mean overlap@{args.k}: {mean_overlap:.4f}")
    print(f"Percentage with overlap ≤ 2: {pct_low_overlap:.2f}%")
    
    # Plot overlap histogram
    plot_overlap_histogram(overlaps, args.model1_name, args.model2_name,
                          save_path=os.path.join(args.save_dir, 'overlap_histogram.png'))
    
    # 6. Compute rank correlation
    print("\n" + "="*70)
    print("STEP 6: COMPUTING RANK CORRELATION")
    print("="*70)
    
    correlations, mean_spearman, std_spearman = compute_rank_correlation(sim_model1, sim_model2)
    
    print(f"Mean Spearman ρ: {mean_spearman:.4f} ± {std_spearman:.4f}")
    
    # 7. Late fusion
    print("\n" + "="*70)
    print("STEP 7: LATE FUSION SANITY CHECK")
    print("="*70)
    
    fusion_results = {}
    
    for alpha in args.fusion_alphas:
        print(f"\nTesting α = {alpha:.1f}...")
        
        sim_fused = late_fusion_retrieval(sim_model1, sim_model2, alpha=alpha)
        topk_fused, scores_fused = retrieve_topk(sim_fused, k=args.k, exclude_self=True)
        map_fused, _ = compute_map_at_k(topk_fused, labels, labels, k=args.k)
        
        fusion_results[alpha] = map_fused
        print(f"  Fusion (α={alpha:.1f}) mAP@{args.k}: {map_fused:.4f}")
    
    # 8. Visualize sample queries
    if args.image_dir and args.visualize_samples > 0:
        print("\n" + "="*70)
        print(f"STEP 8: VISUALIZING {args.visualize_samples} SAMPLE QUERIES")
        print("="*70)
        
        # Select diverse samples (different overlap levels)
        sample_indices = []
        
        # Low overlap (≤ 2)
        low_overlap_queries = np.where(overlaps <= 2)[0]
        if len(low_overlap_queries) > 0:
            sample_indices.append(np.random.choice(low_overlap_queries))
        
        # Medium overlap (3)
        med_overlap_queries = np.where(overlaps == 3)[0]
        if len(med_overlap_queries) > 0:
            sample_indices.append(np.random.choice(med_overlap_queries))
        
        # High overlap (≥ 4)
        high_overlap_queries = np.where(overlaps >= 4)[0]
        if len(high_overlap_queries) > 0:
            sample_indices.append(np.random.choice(high_overlap_queries))
        
        # Fill remaining with random samples
        remaining = args.visualize_samples - len(sample_indices)
        if remaining > 0:
            available = set(range(n_samples)) - set(sample_indices)
            additional = np.random.choice(list(available), size=min(remaining, len(available)), replace=False)
            sample_indices.extend(additional)
        
        for query_idx in sample_indices:
            print(f"Visualizing query {query_idx} (overlap: {overlaps[query_idx]:.0f})...")
            
            visualize_retrieval_comparison(
                query_idx,
                topk_model1[query_idx],
                scores_model1[query_idx],
                topk_model2[query_idx],
                scores_model2[query_idx],
                labels,
                image_ids,
                args.image_dir,
                args.model1_name,
                args.model2_name,
                class_names=args.class_names,
                k=args.k,
                save_path=os.path.join(args.save_dir, f'query_{query_idx}_comparison.png')
            )
    
    # 9. Compile and print summary
    results = {
        'model1_map': map_model1,
        'model2_map': map_model2,
        'mean_overlap': mean_overlap,
        'pct_low_overlap': pct_low_overlap,
        'mean_spearman': mean_spearman,
        'std_spearman': std_spearman,
        'fusion_results': fusion_results,
        'n_queries': n_samples
    }
    
    print_summary_table(results, args.model1_name, args.model2_name)
    
    # Save results
    results_path = os.path.join(args.save_dir, 'backbone_comparison_results.npz')
    np.savez(
        results_path,
        model1_name=args.model1_name,
        model2_name=args.model2_name,
        model1_map=map_model1,
        model2_map=map_model2,
        overlaps=overlaps,
        mean_overlap=mean_overlap,
        correlations=correlations,
        mean_spearman=mean_spearman,
        std_spearman=std_spearman,
        fusion_alphas=list(fusion_results.keys()),
        fusion_maps=list(fusion_results.values()),
        aps_model1=aps_model1,
        aps_model2=aps_model2
    )
    print(f"Results saved to {results_path}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Backbone Comparison for Medical Image Retrieval',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Compare ConvNeXtV2 vs ConceptCLIP
    python test_backbone.py \\
        --model1-embeddings data/convnext_embeddings.npy \\
        --model2-embeddings data/conceptclip_embeddings.npy \\
        --model1-name ConvNeXtV2 --model2-name ConceptCLIP \\
        --labels data/labels.npy \\
        --image-ids data/image_ids.npy \\
        --image-dir /path/to/images \\
        --save-dir results/backbone_comparison \\
        --k 5 \\
        --visualize-samples 5
    
    # Compare MedCLIP vs MedSigLIP
    python test_backbone.py \\
        --model1-embeddings data/medclip_embeddings.npy \\
        --model2-embeddings data/medsiglip_embeddings.npy \\
        --model1-name MedCLIP --model2-name MedSigLIP \\
        --labels data/labels.npy \\
        --save-dir results/medclip_vs_medsiglip
        """
    )
    
    # Required arguments
    parser.add_argument('--model1-embeddings', required=True,
                       help='Path to first model embeddings (.npy)')
    parser.add_argument('--model2-embeddings', required=True,
                       help='Path to second model embeddings (.npy)')
    parser.add_argument('--labels', required=True,
                       help='Path to class labels (.npy)')
    
    # Model names
    parser.add_argument('--model1-name', default='Model1',
                       help='Display name for first model (default: Model1)')
    parser.add_argument('--model2-name', default='Model2',
                       help='Display name for second model (default: Model2)')
    
    # Optional arguments
    parser.add_argument('--image-ids', default=None,
                       help='Path to image IDs (.npy). If not provided, uses indices.')
    parser.add_argument('--image-dir', default=None,
                       help='Directory containing images for visualization')
    parser.add_argument('--save-dir', default='./results/backbone_comparison',
                       help='Directory to save results')
    
    # Evaluation parameters
    parser.add_argument('--k', type=int, default=5,
                       help='Top-k for retrieval evaluation (default: 5)')
    parser.add_argument('--fusion-alphas', type=float, nargs='+',
                       default=[0.3, 0.5, 0.7],
                       help='Alpha values for late fusion (default: 0.3 0.5 0.7)')
    
    # Visualization parameters
    parser.add_argument('--visualize-samples', type=int, default=5,
                       help='Number of sample queries to visualize (default: 5, 0 to disable)')
    parser.add_argument('--class-names', type=str, nargs='+',
                       default=['COVID', 'Pneumonia', 'Normal'],
                       help='Class names in order (default: COVID Pneumonia Normal)')
    
    args = parser.parse_args()
    
    # Convert class names to dict
    args.class_names = {i: name for i, name in enumerate(args.class_names)}
    
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
