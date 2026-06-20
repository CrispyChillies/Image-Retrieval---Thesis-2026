"""
Visualize embeddings as a 2D scatter using UMAP or t-SNE.

Usage examples:
  - From Milvus/Zilliz:
      python visualize_embeddings.py --from-milvus --model_type dinov2 --dataset chestxray \
        --uri <ZILLIZ_URI> --token <ZILLIZ_TOKEN> --limit 2000 --method umap --out viz.png

  - From saved npz file:
      python visualize_embeddings.py --embeddings_file embeddings.npz --method umap --out viz.png

The script outputs a PNG image and (optionally) a CSV of the points.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def fetch_from_milvus(model_type, dataset, uri, token, limit):
    from milvus.milvus_setup import MilvusManager

    manager = MilvusManager(uri=uri, token=token, dataset=dataset)
    if not manager.connect():
        raise RuntimeError("Failed to connect to Milvus/Zilliz")

    manager.load_collection(model_type)
    collection = manager.collections[model_type]

    total = collection.num_entities
    n = min(total, limit) if limit and limit > 0 else total
    print(f"Collection has {total} entities, fetching {n}")

    # Try a simple query to retrieve fields (may be heavy for very large collections)
    try:
        results = collection.query(expr="", output_fields=["image_path", "label", "embedding"], limit=n)
    except Exception:
        # Fall back to retrieving by ranges via query with id filter
        results = []
        step = 1000
        fetched = 0
        while fetched < n:
            batch = collection.query(expr="", output_fields=["image_path", "label", "embedding"], limit=min(step, n - fetched), offset=fetched)
            if not batch:
                break
            results.extend(batch)
            fetched += len(batch)

    image_paths = [r.get("image_path") for r in results]
    labels = [r.get("label") for r in results]
    embeddings = [r.get("embedding") for r in results]

    embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings, labels, image_paths


def load_from_npz(path):
    data = np.load(path, allow_pickle=True)
    embeddings = data.get("embeddings")
    labels = data.get("labels")
    paths = data.get("paths")
    if embeddings is None:
        raise ValueError("No 'embeddings' array found in npz file")
    return embeddings, labels.tolist() if labels is not None else None, paths.tolist() if paths is not None else None


def reduce_dim(embeddings, method="umap", n_components=2, random_state=42, **kwargs):
    if method == "umap":
        try:
            import umap
        except Exception:
            import umap.umap_ as umap  # older package structure
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
        embedding_2d = reducer.fit_transform(embeddings)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=n_components, random_state=random_state, init="pca", learning_rate="auto", **kwargs)
        embedding_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Unknown method: choose 'umap' or 'tsne'")
    return embedding_2d


def plot_embeddings(
    points,
    labels=None,
    paths=None,
    out_path="viz.png",
    title=None,
    marker_size=6,
    alpha=0.8,
    dpi=200,
    cluster_k=None,
    interactive=False,
    zoom=None,
):
    import matplotlib.lines as mlines

    plt.figure(figsize=(10, 10))

    if cluster_k is not None:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=cluster_k, random_state=0)
        cluster_ids = kmeans.fit_predict(points)
        centroids = kmeans.cluster_centers_
    else:
        cluster_ids = None
        centroids = None

    if labels is not None:
        unique_labels = list(sorted(set(labels)))
        color_map = {lab: i for i, lab in enumerate(unique_labels)}
        colors = [color_map.get(l, 0) for l in labels]
        scatter = plt.scatter(points[:, 0], points[:, 1], c=colors, cmap="tab10", s=marker_size, alpha=alpha)

        # If clustering provided, compute majority label per cluster and mark mismatches
        if cluster_ids is not None:
            import numpy as _np
            mismatches = []
            cluster_to_major = {}
            for cid in range(cluster_k):
                inds = [i for i, c in enumerate(cluster_ids) if c == cid]
                if not inds:
                    continue
                labs = [labels[i] for i in inds]
                # majority label
                maj = max(set(labs), key=labs.count)
                cluster_to_major[cid] = maj
                for i in inds:
                    if labels[i] != maj:
                        mismatches.append(i)

            # highlight mismatches with black edge
            if mismatches:
                plt.scatter(points[_np.array(mismatches), 0], points[_np.array(mismatches), 1],
                            facecolors='none', edgecolors='k', s=marker_size*3, linewidths=0.8)

        # legend for labels
        handles = []
        cmap = plt.cm.get_cmap('tab10')
        for lab, idx in color_map.items():
            handles.append(mlines.Line2D([0], [0], marker='o', color='w', label=lab,
                                         markerfacecolor=cmap(idx), markersize=6))
        plt.legend(handles=handles, title="label", loc="best", fontsize=8)
    else:
        plt.scatter(points[:, 0], points[:, 1], s=marker_size, alpha=alpha)

    # plot centroids if present
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=50)

    if zoom is not None:
        xmin, xmax, ymin, ymax = zoom
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    plt.title(title or "Embedding visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    print(f"Saved plot to: {out_path}")

    # Interactive output with Plotly
    if interactive:
        try:
            import plotly.express as px
            import pandas as pd

            df = pd.DataFrame(points, columns=["x", "y"])
            if labels is not None:
                df["label"] = labels
            if paths is not None:
                df["path"] = paths
            fig = px.scatter(df, x="x", y="y", color=("label" if labels is not None else None), hover_data=(['path'] if paths is not None else None), width=1000, height=1000)
            interactive_path = out_path.replace('.png', '.html')
            fig.write_html(interactive_path)
            print(f"Saved interactive HTML to: {interactive_path}")
        except Exception as e:
            print(f"Interactive plotly export failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-milvus", action="store_true", help="Fetch embeddings from Milvus/Zilliz")
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="default")
    parser.add_argument("--uri", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--embeddings_file", type=str, default=None, help=".npz file with arrays: embeddings, labels, paths")
    parser.add_argument("--limit", type=int, default=2000, help="Max number of points to fetch")
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"])
    parser.add_argument("--out", type=str, default="embeddings_viz.png")
    parser.add_argument("--points_csv", type=str, default=None, help="Optional CSV output with 2D coords and metadata")
    args = parser.parse_args()

    if args.from_milvus:
        if not args.model_type:
            raise ValueError("--model_type is required when --from-milvus is set")
        embeddings, labels, paths = fetch_from_milvus(args.model_type, args.dataset, args.uri, args.token, args.limit)
    else:
        if not args.embeddings_file:
            raise ValueError("Provide --embeddings_file or set --from-milvus")
        embeddings, labels, paths = load_from_npz(args.embeddings_file)

    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings loaded")

    print(f"Loaded embeddings: {embeddings.shape}")

    # Random subsample if requested limit is smaller than available
    if args.limit and embeddings.shape[0] > args.limit:
        idx = np.random.RandomState(0).choice(np.arange(embeddings.shape[0]), size=args.limit, replace=False)
        embeddings = embeddings[idx]
        if labels is not None:
            labels = [labels[i] for i in idx]
        if paths is not None:
            paths = [paths[i] for i in idx]

    # Reduce dimensionality
    reducer_kwargs = {"n_neighbors": 15, "min_dist": 0.1} if args.method == "umap" else {}
    points = reduce_dim(embeddings, method=args.method, **reducer_kwargs)

    # Plot
    plot_embeddings(points, labels=labels, paths=paths, out_path=args.out, title=f"{args.model_type or 'embeddings'} ({args.method})")

    # Optionally save CSV with coordinates
    if args.points_csv:
        import csv

        header = ["x", "y"]
        if labels is not None:
            header.append("label")
        if paths is not None:
            header.append("path")

        with open(args.points_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(points.shape[0]):
                row = [float(points[i, 0]), float(points[i, 1])]
                if labels is not None:
                    row.append(labels[i])
                if paths is not None:
                    row.append(paths[i])
                writer.writerow(row)

        print(f"Saved points CSV to: {args.points_csv}")


if __name__ == "__main__":
    main()
