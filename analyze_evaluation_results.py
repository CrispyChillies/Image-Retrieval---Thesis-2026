"""
Analyze results from evaluate_test_dataset_milvus.py
Load JSON results and compute detailed statistics
"""

import json
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os


def load_results(json_file):
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def analyze_results(data, output_dir=None):
    """Analyze and print detailed statistics"""
    
    metadata = data['metadata']
    results = data['results']
    
    print(f"\n{'='*70}")
    print(f"RESULTS ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nConfiguration:")
    print(f"  Model: {metadata['model_type']}")
    print(f"  Explainer: {metadata['explainer']}")
    print(f"  Top-K: {metadata['top_k']}")
    print(f"  Queries: {metadata['num_queries']}")
    print(f"  Timestamp: {metadata['timestamp']}")
    
    # Collect all metrics
    all_del_aucs = []
    all_ins_aucs = []
    all_similarities = []
    all_saliency_stds = []
    
    rank_metrics = defaultdict(lambda: {'del': [], 'ins': [], 'sim': []})
    
    for query_result in results:
        for retrieved in query_result['retrieved']:
            rank = retrieved['rank']
            all_del_aucs.append(retrieved['del_auc'])
            all_ins_aucs.append(retrieved['ins_auc'])
            all_similarities.append(retrieved['similarity'])
            all_saliency_stds.append(retrieved['saliency_std'])
            
            rank_metrics[rank]['del'].append(retrieved['del_auc'])
            rank_metrics[rank]['ins'].append(retrieved['ins_auc'])
            rank_metrics[rank]['sim'].append(retrieved['similarity'])
    
    # Overall statistics
    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*70}")
    
    print(f"\nDeletion AUC:")
    print(f"  Mean: {np.mean(all_del_aucs):.4f}")
    print(f"  Std:  {np.std(all_del_aucs):.4f}")
    print(f"  Min:  {np.min(all_del_aucs):.4f}")
    print(f"  Max:  {np.max(all_del_aucs):.4f}")
    print(f"  Median: {np.median(all_del_aucs):.4f}")
    
    print(f"\nInsertion AUC:")
    print(f"  Mean: {np.mean(all_ins_aucs):.4f}")
    print(f"  Std:  {np.std(all_ins_aucs):.4f}")
    print(f"  Min:  {np.min(all_ins_aucs):.4f}")
    print(f"  Max:  {np.max(all_ins_aucs):.4f}")
    print(f"  Median: {np.median(all_ins_aucs):.4f}")
    
    print(f"\nSimilarity:")
    print(f"  Mean: {np.mean(all_similarities):.4f}")
    print(f"  Std:  {np.std(all_similarities):.4f}")
    print(f"  Min:  {np.min(all_similarities):.4f}")
    print(f"  Max:  {np.max(all_similarities):.4f}")
    
    print(f"\nSaliency Std Dev:")
    print(f"  Mean: {np.mean(all_saliency_stds):.4f}")
    print(f"  Min:  {np.min(all_saliency_stds):.4f}")
    print(f"  Max:  {np.max(all_saliency_stds):.4f}")
    
    # Per-rank statistics
    print(f"\n{'='*70}")
    print(f"PER-RANK STATISTICS")
    print(f"{'='*70}")
    
    print(f"\n{'Rank':<6} {'Similarity':<12} {'DEL AUC':<12} {'INS AUC':<12} {'Count':<8}")
    print('-' * 70)
    
    for rank in sorted(rank_metrics.keys()):
        metrics = rank_metrics[rank]
        print(f"{rank:<6} "
              f"{np.mean(metrics['sim']):.4f}±{np.std(metrics['sim']):.4f}  "
              f"{np.mean(metrics['del']):.4f}±{np.std(metrics['del']):.4f}  "
              f"{np.mean(metrics['ins']):.4f}±{np.std(metrics['ins']):.4f}  "
              f"{len(metrics['del']):<8}")
    
    # Quality metrics
    print(f"\n{'='*70}")
    print(f"QUALITY ASSESSMENT")
    print(f"{'='*70}")
    
    good_del = sum(1 for auc in all_del_aucs if auc < 0.6)
    excellent_del = sum(1 for auc in all_del_aucs if auc < 0.5)
    poor_del = sum(1 for auc in all_del_aucs if auc > 0.7)
    
    good_ins = sum(1 for auc in all_ins_aucs if auc > 0.7)
    excellent_ins = sum(1 for auc in all_ins_aucs if auc > 0.8)
    poor_ins = sum(1 for auc in all_ins_aucs if auc < 0.6)
    
    total = len(all_del_aucs)
    
    print(f"\nDeletion Quality:")
    print(f"  Excellent (< 0.5): {excellent_del:4d} ({100*excellent_del/total:5.1f}%)")
    print(f"  Good (< 0.6):      {good_del:4d} ({100*good_del/total:5.1f}%)")
    print(f"  Poor (> 0.7):      {poor_del:4d} ({100*poor_del/total:5.1f}%)")
    
    print(f"\nInsertion Quality:")
    print(f"  Excellent (> 0.8): {excellent_ins:4d} ({100*excellent_ins/total:5.1f}%)")
    print(f"  Good (> 0.7):      {good_ins:4d} ({100*good_ins/total:5.1f}%)")
    print(f"  Poor (< 0.6):      {poor_ins:4d} ({100*poor_ins/total:5.1f}%)")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    avg_del = np.mean(all_del_aucs)
    avg_ins = np.mean(all_ins_aucs)
    
    if avg_del < 0.5 and avg_ins > 0.8:
        print(f"  ✅ EXCELLENT: Very high quality saliency maps")
    elif avg_del < 0.6 and avg_ins > 0.7:
        print(f"  ✅ GOOD: High quality saliency maps")
    elif avg_del < 0.7 and avg_ins > 0.6:
        print(f"  ⚠️  MODERATE: Acceptable saliency quality")
    else:
        print(f"  ❌ POOR: Low quality saliency maps")
        if avg_del > 0.7:
            print(f"     - High deletion AUC ({avg_del:.4f}) - removing salient pixels doesn't reduce similarity enough")
        if avg_ins < 0.6:
            print(f"     - Low insertion AUC ({avg_ins:.4f}) - adding salient pixels doesn't increase similarity enough")
    
    # Create visualizations if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        create_visualizations(all_del_aucs, all_ins_aucs, all_similarities, 
                            rank_metrics, metadata, output_dir)
    
    return {
        'del_auc_mean': float(np.mean(all_del_aucs)),
        'del_auc_std': float(np.std(all_del_aucs)),
        'ins_auc_mean': float(np.mean(all_ins_aucs)),
        'ins_auc_std': float(np.std(all_ins_aucs)),
        'similarity_mean': float(np.mean(all_similarities)),
        'similarity_std': float(np.std(all_similarities))
    }


def create_visualizations(del_aucs, ins_aucs, similarities, rank_metrics, metadata, output_dir):
    """Create visualization plots"""
    
    # 1. Distribution histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(del_aucs, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0].axvline(np.mean(del_aucs), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(del_aucs):.4f}')
    axes[0].set_xlabel('Deletion AUC')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Deletion AUC Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(ins_aucs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1].axvline(np.mean(ins_aucs), color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ins_aucs):.4f}')
    axes[1].set_xlabel('Insertion AUC')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Insertion AUC Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(similarities, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[2].axvline(np.mean(similarities), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {np.mean(similarities):.4f}')
    axes[2].set_xlabel('Cosine Similarity')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Similarity Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'distributions.png')}")
    
    # 2. Per-rank comparison
    ranks = sorted(rank_metrics.keys())
    del_means = [np.mean(rank_metrics[r]['del']) for r in ranks]
    del_stds = [np.std(rank_metrics[r]['del']) for r in ranks]
    ins_means = [np.mean(rank_metrics[r]['ins']) for r in ranks]
    ins_stds = [np.std(rank_metrics[r]['ins']) for r in ranks]
    sim_means = [np.mean(rank_metrics[r]['sim']) for r in ranks]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].errorbar(ranks, del_means, yerr=del_stds, marker='o', capsize=5, label='Deletion', color='red')
    axes[0].errorbar(ranks, ins_means, yerr=ins_stds, marker='s', capsize=5, label='Insertion', color='blue')
    axes[0].set_xlabel('Retrieval Rank')
    axes[0].set_ylabel('AUC')
    axes[0].set_title('Insertion/Deletion AUC by Rank')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(ranks)
    
    axes[1].plot(ranks, sim_means, marker='o', color='green', linewidth=2)
    axes[1].set_xlabel('Retrieval Rank')
    axes[1].set_ylabel('Similarity')
    axes[1].set_title('Average Similarity by Rank')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(ranks)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rank_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'rank_comparison.png')}")
    
    # 3. Scatter plot: DEL vs INS
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(del_aucs, ins_aucs, alpha=0.5, c=similarities, cmap='viridis', s=10)
    ax.set_xlabel('Deletion AUC')
    ax.set_ylabel('Insertion AUC')
    ax.set_title('Deletion vs Insertion AUC\n(color = similarity)')
    ax.axvline(0.6, color='red', linestyle='--', alpha=0.5, label='DEL threshold (0.6)')
    ax.axhline(0.7, color='blue', linestyle='--', alpha=0.5, label='INS threshold (0.7)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Similarity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'del_vs_ins.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'del_vs_ins.png')}")


def compare_multiple_results(json_files, labels=None, output_dir=None):
    """Compare results from multiple experiments"""
    
    if labels is None:
        labels = [f"Exp{i+1}" for i in range(len(json_files))]
    
    all_data = []
    for json_file in json_files:
        data = load_results(json_file)
        all_data.append(data)
    
    print(f"\n{'='*70}")
    print(f"COMPARISON OF {len(json_files)} EXPERIMENTS")
    print(f"{'='*70}")
    
    print(f"\n{'Experiment':<20} {'Model':<15} {'Explainer':<10} {'DEL AUC':<12} {'INS AUC':<12}")
    print('-' * 70)
    
    for label, data in zip(labels, all_data):
        metadata = data['metadata']
        results = data['results']
        
        all_del = []
        all_ins = []
        for query_result in results:
            for retrieved in query_result['retrieved']:
                all_del.append(retrieved['del_auc'])
                all_ins.append(retrieved['ins_auc'])
        
        print(f"{label:<20} {metadata['model_type']:<15} {metadata['explainer']:<10} "
              f"{np.mean(all_del):.4f}±{np.std(all_del):.3f} "
              f"{np.mean(all_ins):.4f}±{np.std(all_ins):.3f}")
    
    # Create comparison plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        del_data = []
        ins_data = []
        
        for data in all_data:
            results = data['results']
            all_del = [r['del_auc'] for query_result in results for r in query_result['retrieved']]
            all_ins = [r['ins_auc'] for query_result in results for r in query_result['retrieved']]
            del_data.append(all_del)
            ins_data.append(all_ins)
        
        bp1 = axes[0].boxplot(del_data, labels=labels, patch_artist=True)
        axes[0].set_ylabel('Deletion AUC')
        axes[0].set_title('Deletion AUC Comparison')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        bp2 = axes[1].boxplot(ins_data, labels=labels, patch_artist=True)
        axes[1].set_ylabel('Insertion AUC')
        axes[1].set_title('Insertion AUC Comparison')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Color boxes
        for patch in bp1['boxes']:
            patch.set_facecolor('lightcoral')
        for patch in bp2['boxes']:
            patch.set_facecolor('lightblue')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved comparison plot: {os.path.join(output_dir, 'comparison.png')}")


def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument('--results', type=str, required=True, nargs='+',
                       help='JSON result file(s) to analyze')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Labels for comparison (if multiple results)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualization plots')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple result files')
    
    args = parser.parse_args()
    
    if args.compare and len(args.results) > 1:
        compare_multiple_results(args.results, args.labels, args.output_dir)
    elif len(args.results) == 1:
        data = load_results(args.results[0])
        analyze_results(data, args.output_dir)
    else:
        print("Provide --compare flag for multiple files or single file for analysis")


if __name__ == '__main__':
    main()
