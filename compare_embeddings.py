import argparse
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    from sklearn.manifold import TSNE
    UMAP_AVAILABLE = False


def guess_key(npz, candidates):
    for k in candidates:
        if k in npz:
            return k
    return None


def load_embeddings(path):
    npz = np.load(path, allow_pickle=True)
    embed_key = guess_key(npz, ["embeds", "embeddings", "image_features", "features"])
    label_key = guess_key(npz, ["labels", "image_labels", "y"])
    if embed_key is None:
        raise ValueError(f"No embedding key found in {path}. Available: {list(npz.keys())}")
    embeds = npz[embed_key]
    labels = npz[label_key] if label_key is not None else None
    return embeds, labels, embed_key, label_key


def stats(embeds):
    s = {}
    s['shape'] = embeds.shape
    s['dtype'] = embeds.dtype
    s['mean'] = float(np.mean(embeds))
    s['std'] = float(np.std(embeds))
    norms = np.linalg.norm(embeds, axis=1)
    s['norm_mean'] = float(np.mean(norms))
    s['norm_std'] = float(np.std(norms))
    return s


def pairwise_cosine_sample(embeds, n_samples=10000):
    rng = np.random.default_rng(0)
    n = embeds.shape[0]
    idx_a = rng.integers(0, n, size=n_samples)
    idx_b = rng.integers(0, n, size=n_samples)
    va = embeds[idx_a]
    vb = embeds[idx_b]
    # normalize
    va = va / (np.linalg.norm(va, axis=1, keepdims=True) + 1e-10)
    vb = vb / (np.linalg.norm(vb, axis=1, keepdims=True) + 1e-10)
    cos = np.sum(va * vb, axis=1)
    return cos


def reduce_for_plot(embeds, n_components=2, n_neighbors=15, min_dist=0.1):
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=n_components, random_state=0, n_neighbors=n_neighbors, min_dist=min_dist)
        proj = reducer.fit_transform(embeds)
    else:
        tsne = TSNE(n_components=n_components, init='pca', random_state=0)
        proj = tsne.fit_transform(embeds)
    return proj


def plot_side_by_side(a_proj, b_proj, a_labels, b_labels, out_png, titles=("A","B")):
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    for ax, proj, labels, title in zip(axes, [a_proj, b_proj], [a_labels, b_labels], titles):
        if labels is None:
            ax.scatter(proj[:,0], proj[:,1], s=6, alpha=0.7)
        else:
            labels = np.array(labels)
            unique = np.unique(labels)
            cmap = plt.get_cmap('tab10')
            for i, u in enumerate(unique):
                mask = labels==u
                ax.scatter(proj[mask,0], proj[mask,1], s=8, alpha=0.8, label=str(u), c=[cmap(i%10)])
            ax.legend(markerscale=2, fontsize='small')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_a', required=True, help='First .npz of embeddings')
    parser.add_argument('--file_b', required=True, help='Second .npz of embeddings')
    parser.add_argument('--out', default='compare_embeddings', help='Output prefix')
    parser.add_argument('--sample_pairs', type=int, default=5000, help='Number of random pairs for cosine sample')
    args = parser.parse_args()

    a_path = Path(args.file_a)
    b_path = Path(args.file_b)
    assert a_path.exists(), f"{a_path} not found"
    assert b_path.exists(), f"{b_path} not found"

    a_emb, a_labels, a_ek, a_lk = load_embeddings(str(a_path))
    b_emb, b_labels, b_ek, b_lk = load_embeddings(str(b_path))

    report_lines = []
    report_lines.append(f"File A: {a_path} (embeds key={a_ek}, labels key={a_lk})")
    report_lines.append(f"File B: {b_path} (embeds key={b_ek}, labels key={b_lk})")
    report_lines.append("")

    a_stats = stats(a_emb)
    b_stats = stats(b_emb)

    report_lines.append("A stats:")
    for k,v in a_stats.items():
        report_lines.append(f"  {k}: {v}")
    report_lines.append("")
    report_lines.append("B stats:")
    for k,v in b_stats.items():
        report_lines.append(f"  {k}: {v}")
    report_lines.append("")

    # If same shape, compare elementwise cosine between corresponding rows
    if a_emb.shape == b_emb.shape:
        # compute mean cosine between corresponding vectors
        a_norm = a_emb / (np.linalg.norm(a_emb, axis=1, keepdims=True) + 1e-10)
        b_norm = b_emb / (np.linalg.norm(b_emb, axis=1, keepdims=True) + 1e-10)
        cor_cos = np.sum(a_norm * b_norm, axis=1)
        report_lines.append(f"Mean cosine between corresponding embeddings: {float(np.mean(cor_cos)):.4f}")
        report_lines.append(f"Std cosine between corresponding embeddings: {float(np.std(cor_cos)):.4f}")
    else:
        report_lines.append("Embeddings shapes differ; skipping per-sample cosine comparison.")

    # Pairwise cosine samples
    report_lines.append("")
    a_cos = pairwise_cosine_sample(a_emb, n_samples=args.sample_pairs)
    b_cos = pairwise_cosine_sample(b_emb, n_samples=args.sample_pairs)
    report_lines.append(f"A random pairwise cosine mean: {float(np.mean(a_cos)):.4f}, std: {float(np.std(a_cos)):.4f}")
    report_lines.append(f"B random pairwise cosine mean: {float(np.mean(b_cos)):.4f}, std: {float(np.std(b_cos)):.4f}")

    # If labels provided, compute intra-class vs inter-class mean cosine
    def intra_inter_stats(emb, labels):
        if labels is None:
            return None
        labels = np.array(labels)
        unique = np.unique(labels)
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        intra = []
        inter = []
        for i,u in enumerate(unique):
            idx = np.where(labels==u)[0]
            if len(idx) < 2:
                continue
            # sample up to 1000 pairs within class
            import itertools
            pairs = np.array(list(itertools.combinations(idx, 2)))
            if pairs.shape[0] > 1000:
                rng = np.random.default_rng(0)
                sel = rng.choice(pairs.shape[0], size=1000, replace=False)
                pairs = pairs[sel]
            ia = emb_n[pairs[:,0]]
            ib = emb_n[pairs[:,1]]
            intra.extend((ia * ib).sum(axis=1).tolist())
        # inter-class: sample random pairs with different labels
        rng = np.random.default_rng(1)
        n = emb.shape[0]
        n_samples = min(2000, n*(n-1)//2)
        for _ in range(n_samples):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if labels[i] != labels[j]:
                inter.append(float(np.dot(emb_n[i], emb_n[j])))
        if len(intra)==0 or len(inter)==0:
            return None
        return {
            'intra_mean': float(np.mean(intra)),
            'intra_std': float(np.std(intra)),
            'inter_mean': float(np.mean(inter)),
            'inter_std': float(np.std(inter)),
        }

    a_ii = intra_inter_stats(a_emb, a_labels)
    b_ii = intra_inter_stats(b_emb, b_labels)
    if a_ii is not None:
        report_lines.append("")
        report_lines.append("A intra/inter class cosine:")
        for k,v in a_ii.items():
            report_lines.append(f"  {k}: {v}")
    if b_ii is not None:
        report_lines.append("")
        report_lines.append("B intra/inter class cosine:")
        for k,v in b_ii.items():
            report_lines.append(f"  {k}: {v}")

    # Save report
    out_prefix = args.out
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
    rpt_path = out_prefix + '_report.txt'
    with open(rpt_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print('\n'.join(report_lines))
    print(f'\nReport saved to {rpt_path}')

    # Create 2D projections (sample if too large)
    max_plot = 2000
    def sample_for_plot(emb, labels):
        n = emb.shape[0]
        if n > max_plot:
            rng = np.random.default_rng(0)
            sel = rng.choice(n, size=max_plot, replace=False)
            return emb[sel], (labels[sel] if labels is not None else None)
        return emb, labels

    a_plot_emb, a_plot_labels = sample_for_plot(a_emb, a_labels)
    b_plot_emb, b_plot_labels = sample_for_plot(b_emb, b_labels)

    a_proj = reduce_for_plot(a_plot_emb)
    b_proj = reduce_for_plot(b_plot_emb)

    out_png = out_prefix + '_umap_compare.png'
    plot_side_by_side(a_proj, b_proj, a_plot_labels, b_plot_labels, out_png, titles=(f"A: {a_path.name}", f"B: {b_path.name}"))
    print(f'Projection image saved to {out_png}')


if __name__ == '__main__':
    main()
