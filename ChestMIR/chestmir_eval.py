from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from pymilvus import Collection, connections


DEFAULT_LESIONS = [
    "Consolidation",
    "Lung Opacity",
    "Infiltration",
    "Atelectasis",
    "Pleural effusion",
]

LESION_ALIAS_GROUPS = {
    "consolidation": [
        "consolidation",
    ],
    "lung opacity": [
        "lung opacity",
        "lung_opacity",
        "opacity",
        "opacities",
    ],
    "infiltration": [
        "infiltration",
        "infiltrate",
        "infiltrates",
    ],
    "atelectasis": [
        "atelectasis",
        "atelectatic",
    ],
    "pleural effusion": [
        "pleural effusion",
        "pleural_effusion",
        "effusion",
        "plural effusion",
    ],
}

LESION_ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _aliases in LESION_ALIAS_GROUPS.items():
    for _alias in _aliases:
        LESION_ALIAS_TO_CANON[_alias] = _canon


@dataclass
class EvalDataset:
    image_names: list[str]
    labels: np.ndarray  # dtype=object, shape [N]
    global_vectors: np.ndarray  # shape [N, D]
    lesion_vectors: list[dict[str, list[np.ndarray]]]


def compute_ap(ranks: np.ndarray, nres: int) -> float:
    """Compute average precision for ranked positive indexes."""
    nimgranks = len(ranks)
    ap = 0.0
    recall_step = 1.0 / nres

    for j in np.arange(nimgranks):
        rank = int(ranks[j])
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0

    return float(ap)


def compute_map(ranks: np.ndarray, gnd: np.ndarray, kappas: list[int] | tuple[int, ...] = ()):
    """Exact mAP/mP@K implementation aligned with test.py behavior."""
    mAP = 0.0
    nq = len(gnd)
    aps = np.zeros(nq, dtype=np.float64)
    pr = np.zeros(len(kappas), dtype=np.float64)
    prs = np.zeros((nq, len(kappas)), dtype=np.float64)
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.where(gnd == gnd[i])[0]
        if qgnd.shape[0] == 0:
            aps[i] = float("nan")
            prs[i, :] = float("nan")
            nempty += 1
            continue

        pos = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgnd)]
        ap = compute_ap(pos, len(qgnd))
        mAP += ap
        aps[i] = ap

        pos = pos + 1  # 1-based
        for j in np.arange(len(kappas)):
            kq = min(int(max(pos)), int(kappas[j]))
            prs[i, j] = (pos <= kq).sum() / kq
        pr += prs[i, :]

    denom = max(1, nq - nempty)
    mAP = mAP / denom
    pr = pr / denom
    return mAP, aps, pr, prs


def majority_vote(retrieved_labels: np.ndarray) -> Any:
    if len(retrieved_labels) == 0:
        return None
    counter = Counter(retrieved_labels.tolist())
    return counter.most_common(1)[0][0]


def compute_classification_metrics_from_ranks(
    labels: np.ndarray,
    ranks: np.ndarray,
    k_values: list[int],
) -> dict[int, dict[str, float]]:
    """Compute majority-vote classification metrics from ranking matrix."""
    n_samples = len(labels)
    results: dict[int, dict[str, float]] = {}

    for k in k_values:
        predicted_labels: list[Any] = []
        true_labels: list[Any] = []

        for i in range(n_samples):
            top_k_indices = ranks[:k, i]
            retrieved_labels = labels[top_k_indices]
            pred_label = majority_vote(retrieved_labels)
            predicted_labels.append(pred_label)
            true_labels.append(labels[i])

        y_true = np.asarray(true_labels, dtype=object)
        y_pred = np.asarray(predicted_labels, dtype=object)
        classes = np.unique(np.concatenate([y_true, y_pred], axis=0))

        per_class_p: list[float] = []
        per_class_r: list[float] = []
        per_class_f1: list[float] = []
        supports: list[int] = []

        for c in classes:
            tp = int(np.sum((y_true == c) & (y_pred == c)))
            fp = int(np.sum((y_true != c) & (y_pred == c)))
            fn = int(np.sum((y_true == c) & (y_pred != c)))
            support = int(np.sum(y_true == c))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
            per_class_p.append(p)
            per_class_r.append(r)
            per_class_f1.append(f1)
            supports.append(support)

        supports_np = np.asarray(supports, dtype=np.float64)
        weight_sum = float(np.sum(supports_np))
        if weight_sum <= 0:
            weight_sum = 1.0
        weights = supports_np / weight_sum

        precision_macro = float(np.mean(per_class_p)) if per_class_p else 0.0
        recall_macro = float(np.mean(per_class_r)) if per_class_r else 0.0
        f1_macro = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

        precision_weighted = float(np.sum(np.asarray(per_class_p) * weights)) if per_class_p else 0.0
        recall_weighted = float(np.sum(np.asarray(per_class_r) * weights)) if per_class_r else 0.0
        f1_weighted = float(np.sum(np.asarray(per_class_f1) * weights)) if per_class_f1 else 0.0

        acc = float(np.mean(y_true == y_pred))

        results[k] = {
            "accuracy": acc * 100.0,
            "precision_macro": precision_macro * 100.0,
            "recall_macro": recall_macro * 100.0,
            "f1_macro": f1_macro * 100.0,
            "precision_weighted": precision_weighted * 100.0,
            "recall_weighted": recall_weighted * 100.0,
            "f1_weighted": f1_weighted * 100.0,
        }

    return results


def retrieval_accuracy_from_ranks(ranks: np.ndarray, labels: np.ndarray, topk: list[int] | tuple[int, ...]):
    n = len(labels)
    out: list[float] = []
    for k in topk:
        correct = 0
        for i in range(n):
            top_idx = ranks[:k, i]
            if np.any(labels[top_idx] == labels[i]):
                correct += 1
        out.append((correct * 100.0) / max(1, n))
    return np.array(out, dtype=np.float64)


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def parse_json_list(raw: str) -> list[Any]:
    if raw is None or raw == "":
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except Exception:
        return []


def _normalize_lesion_text(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = " ".join(text.split())
    return text


def canonical_lesion_name(name: str) -> str:
    normalized = _normalize_lesion_text(name)
    return LESION_ALIAS_TO_CANON.get(normalized, normalized)


def build_lesion_vector_map(region_labels_json: str, region_vectors_json: str) -> dict[str, list[np.ndarray]]:
    labels = parse_json_list(region_labels_json)
    vectors = parse_json_list(region_vectors_json)
    n = min(len(labels), len(vectors))
    out: dict[str, list[np.ndarray]] = {}

    for i in range(n):
        lesion = canonical_lesion_name(labels[i])
        vec_raw = vectors[i]
        if not isinstance(vec_raw, list) or len(vec_raw) == 0:
            continue
        vec = np.asarray(vec_raw, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm <= 0:
            continue
        vec = vec / norm
        out.setdefault(lesion, []).append(vec)

    return out


def load_eval_dataset(
    host: str,
    port: int,
    collection_name: str,
    fetch_batch_size: int = 2048,
) -> EvalDataset:
    connections.connect(alias="default", host=host, port=str(port))
    collection = Collection(collection_name)
    collection.load()

    total = int(collection.num_entities)
    if total <= 1:
        raise ValueError(f"Collection '{collection_name}' has too few entities: {total}")

    fields = ["image_name", "label", "global_vector", "region_labels_json", "region_vectors_json"]
    rows: list[dict[str, Any]] = []

    offset = 0
    while offset < total:
        chunk = collection.query(
            expr="id >= 0",
            output_fields=fields,
            limit=min(fetch_batch_size, total - offset),
            offset=offset,
        )
        if not chunk:
            break
        rows.extend(chunk)
        offset += len(chunk)

    image_names: list[str] = []
    labels: list[str] = []
    vectors: list[np.ndarray] = []
    lesion_maps: list[dict[str, list[np.ndarray]]] = []

    for r in rows:
        g = np.asarray(r["global_vector"], dtype=np.float32)
        if g.ndim != 1 or g.size == 0:
            continue
        image_names.append(str(r.get("image_name", "")))
        labels.append(str(r.get("label", "unknown")))
        vectors.append(g)
        lesion_maps.append(
            build_lesion_vector_map(
                str(r.get("region_labels_json", "[]")),
                str(r.get("region_vectors_json", "[]")),
            )
        )

    if len(vectors) <= 1:
        raise ValueError("Insufficient valid vectors after parsing from Milvus")

    global_vectors = np.stack(vectors, axis=0)
    global_vectors = normalize_rows(global_vectors)

    return EvalDataset(
        image_names=image_names,
        labels=np.asarray(labels, dtype=object),
        global_vectors=global_vectors,
        lesion_vectors=lesion_maps,
    )


def similarity_to_ranks(sim: np.ndarray) -> np.ndarray:
    # returns shape [N, N], where column i is ranking list for query i
    return np.argsort(-sim, axis=0)


def evaluate_rankings(
    ranks: np.ndarray,
    labels: np.ndarray,
    kappas: list[int],
    cls_k_values: list[int],
) -> dict[str, Any]:
    acc = retrieval_accuracy_from_ranks(ranks, labels, kappas)
    mAP, _aps, pr, _prs = compute_map(ranks, labels, kappas)
    cls = compute_classification_metrics_from_ranks(labels, ranks, cls_k_values)
    return {
        "R@K": {k: float(v) for k, v in zip(kappas, acc)},
        "mAP": float(mAP * 100.0),
        "mP@K": {k: float(v * 100.0) for k, v in zip(kappas, pr)},
        "classification": cls,
    }


def choose_query_lesion_vector(
    lesion_map: dict[str, list[np.ndarray]],
    lesion_name: str,
) -> np.ndarray | None:
    key = canonical_lesion_name(lesion_name)
    cands = lesion_map.get(key, [])
    if not cands:
        return None
    return cands[0]


def best_candidate_lesion_score(
    query_vec: np.ndarray,
    candidate_lesions: dict[str, list[np.ndarray]],
    lesion_name: str,
) -> float:
    key = canonical_lesion_name(lesion_name)
    cands = candidate_lesions.get(key, [])
    if not cands:
        return -1.0
    scores = [float(np.dot(query_vec, c)) for c in cands]
    return max(scores) if scores else -1.0


def choose_query_adaptive_lesion_vector(
    lesion_map: dict[str, list[np.ndarray]],
    target_lesions: list[str],
) -> tuple[str | None, np.ndarray | None]:
    target_keys = [canonical_lesion_name(x) for x in target_lesions]
    target_key_set = set(target_keys)

    best_name: str | None = None
    best_vec: np.ndarray | None = None
    best_count = -1

    for lesion_name in target_keys:
        cands = lesion_map.get(lesion_name, [])
        if not cands:
            continue
        count = len(cands)
        if count > best_count:
            best_count = count
            best_name = lesion_name
            best_vec = cands[0]

    if best_name is not None and best_vec is not None:
        return best_name, best_vec

    for lesion_name, cands in lesion_map.items():
        if lesion_name not in target_key_set or not cands:
            continue
        return lesion_name, cands[0]

    return None, None


def rerank_with_specific_lesion(
    base_sim: np.ndarray,
    lesion_maps: list[dict[str, list[np.ndarray]]],
    lesion_name: str,
    rerank_topk: int,
    global_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = base_sim.shape[0]
    ranks_base = similarity_to_ranks(base_sim)
    ranks_new = np.empty_like(ranks_base)

    fallback_queries = 0
    reranked_queries = 0

    for i in range(n):
        base_rank = ranks_base[:, i]
        topk = min(rerank_topk, n - 1)
        top_idx = base_rank[:topk]

        q_vec = choose_query_lesion_vector(lesion_maps[i], lesion_name)
        if q_vec is None:
            fallback_queries += 1
            ranks_new[:, i] = base_rank
            continue

        reranked_queries += 1
        combined_scores: list[tuple[int, float, float]] = []
        for j in top_idx:
            region_score = best_candidate_lesion_score(q_vec, lesion_maps[j], lesion_name)
            # still cosine-based; combine global cosine + lesion cosine for reranking
            score = (global_weight * float(base_sim[j, i])) + ((1.0 - global_weight) * region_score)
            combined_scores.append((int(j), score, float(base_sim[j, i])))

        combined_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        new_top = [x[0] for x in combined_scores]

        in_top = np.zeros(n, dtype=bool)
        in_top[new_top] = True
        tail = [idx for idx in base_rank if not in_top[idx]]
        ranks_new[:, i] = np.asarray(new_top + tail, dtype=np.int64)

    stats = {
        "lesion": lesion_name,
        "queries_total": n,
        "queries_reranked": reranked_queries,
        "queries_fallback_global": fallback_queries,
        "rerank_topk": rerank_topk,
        "global_weight": global_weight,
        "region_weight": 1.0 - global_weight,
    }
    return ranks_new, stats


def rerank_with_adaptive_lesion(
    base_sim: np.ndarray,
    lesion_maps: list[dict[str, list[np.ndarray]]],
    target_lesions: list[str],
    rerank_topk: int,
    global_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = base_sim.shape[0]
    ranks_base = similarity_to_ranks(base_sim)
    ranks_new = np.empty_like(ranks_base)

    fallback_queries = 0
    reranked_queries = 0
    lesion_usage: Counter[str] = Counter()

    for i in range(n):
        base_rank = ranks_base[:, i]
        topk = min(rerank_topk, n - 1)
        top_idx = base_rank[:topk]

        chosen_lesion, q_vec = choose_query_adaptive_lesion_vector(lesion_maps[i], target_lesions)
        if q_vec is None or chosen_lesion is None:
            fallback_queries += 1
            ranks_new[:, i] = base_rank
            continue

        lesion_usage[chosen_lesion] += 1
        reranked_queries += 1

        combined_scores: list[tuple[int, float, float]] = []
        for j in top_idx:
            region_score = best_candidate_lesion_score(q_vec, lesion_maps[j], chosen_lesion)
            score = (global_weight * float(base_sim[j, i])) + ((1.0 - global_weight) * region_score)
            combined_scores.append((int(j), score, float(base_sim[j, i])))

        combined_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        new_top = [x[0] for x in combined_scores]

        in_top = np.zeros(n, dtype=bool)
        in_top[new_top] = True
        tail = [idx for idx in base_rank if not in_top[idx]]
        ranks_new[:, i] = np.asarray(new_top + tail, dtype=np.int64)

    stats = {
        "mode": "adaptive",
        "queries_total": n,
        "queries_reranked": reranked_queries,
        "queries_fallback_global": fallback_queries,
        "rerank_topk": rerank_topk,
        "global_weight": global_weight,
        "region_weight": 1.0 - global_weight,
        "lesion_usage": dict(lesion_usage),
    }
    return ranks_new, stats


def print_stage_report(title: str, report: dict[str, Any], kappas: list[int], cls_k_values: list[int]) -> None:
    print(f"\n=== {title} ===")
    rline = ", ".join([f"R@{k}: {report['R@K'][k]:.2f}%" for k in kappas])
    pline = ", ".join([f"P@{k}: {report['mP@K'][k]:.2f}%" for k in kappas])
    print(rline)
    print(f"mAP: {report['mAP']:.2f}%")
    print(pline)

    for k in cls_k_values:
        m = report["classification"][k]
        print(
            f"Top-{k}: Acc {m['accuracy']:.2f}% | "
            f"P_macro {m['precision_macro']:.2f}% | R_macro {m['recall_macro']:.2f}% | "
            f"F1_macro {m['f1_macro']:.2f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="ChestMIR two-stage evaluation from Milvus")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19530)
    parser.add_argument("--collection-name", type=str, default="covid_test_mir")
    parser.add_argument("--kappas", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--classification-k", type=int, nargs="+", default=[1, 5, 10, 15, 20])
    parser.add_argument("--rerank-topk", type=int, default=50)
    parser.add_argument(
        "--lesions",
        type=str,
        nargs="+",
        default=DEFAULT_LESIONS,
        help="Target lesions for stage-2 reranking",
    )
    parser.add_argument(
        "--global-weight",
        type=float,
        default=0.5,
        help="Weight of global cosine in stage-2 score (region weight = 1-global_weight)",
    )
    args = parser.parse_args()

    if not (0.0 <= args.global_weight <= 1.0):
        raise ValueError("--global-weight must be in [0, 1]")

    dataset: EvalDataset | None = None
    try:
        dataset = load_eval_dataset(args.host, args.port, args.collection_name)

        sim_global = dataset.global_vectors @ dataset.global_vectors.T
        np.fill_diagonal(sim_global, -np.inf)

        ranks_stage1 = similarity_to_ranks(sim_global)
        report_stage1 = evaluate_rankings(
            ranks_stage1,
            dataset.labels,
            kappas=args.kappas,
            cls_k_values=args.classification_k,
        )
        print_stage_report("Stage 1 - Global Retrieval", report_stage1, args.kappas, args.classification_k)

        ranks_adaptive, adaptive_stats = rerank_with_adaptive_lesion(
            base_sim=sim_global,
            lesion_maps=dataset.lesion_vectors,
            target_lesions=args.lesions,
            rerank_topk=args.rerank_topk,
            global_weight=args.global_weight,
        )
        report_adaptive = evaluate_rankings(
            ranks_adaptive,
            dataset.labels,
            kappas=args.kappas,
            cls_k_values=args.classification_k,
        )
        print_stage_report("Stage 2 - Adaptive Lesion Rerank", report_adaptive, args.kappas, args.classification_k)
        print(
            f"Fallback(global-only): {adaptive_stats['queries_fallback_global']}/{adaptive_stats['queries_total']} | "
            f"Reranked: {adaptive_stats['queries_reranked']}/{adaptive_stats['queries_total']} | "
            f"topK={adaptive_stats['rerank_topk']}"
        )
        if adaptive_stats["lesion_usage"]:
            usage = ", ".join([f"{k}:{v}" for k, v in sorted(adaptive_stats["lesion_usage"].items())])
            print(f"Adaptive lesion usage: {usage}")

        lesion_reports: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for lesion in args.lesions:
            ranks_rerank, stats = rerank_with_specific_lesion(
                base_sim=sim_global,
                lesion_maps=dataset.lesion_vectors,
                lesion_name=lesion,
                rerank_topk=args.rerank_topk,
                global_weight=args.global_weight,
            )

            report = evaluate_rankings(
                ranks_rerank,
                dataset.labels,
                kappas=args.kappas,
                cls_k_values=args.classification_k,
            )
            lesion_reports.append(report)
            summary_rows.append(
                {
                    "lesion": lesion,
                    "mAP": report["mAP"],
                    "R@1": report["R@K"][args.kappas[0]],
                    "R@5": report["R@K"][5] if 5 in report["R@K"] else np.nan,
                    "fallback": stats["queries_fallback_global"],
                    "reranked": stats["queries_reranked"],
                }
            )

            print_stage_report(f"Stage 2 - Lesion Rerank ({lesion})", report, args.kappas, args.classification_k)
            print(
                f"Fallback(global-only): {stats['queries_fallback_global']}/{stats['queries_total']} | "
                f"Reranked: {stats['queries_reranked']}/{stats['queries_total']} | "
                f"topK={stats['rerank_topk']}"
            )

        if summary_rows:
            mean_map = float(np.mean([r["mAP"] for r in summary_rows]))
            mean_r1 = float(np.mean([r["R@1"] for r in summary_rows]))
            valid_r5 = [r["R@5"] for r in summary_rows if not np.isnan(r["R@5"])]
            mean_r5 = float(np.mean(valid_r5)) if valid_r5 else float("nan")

            print("\n=== Final Summary Across 5 Lesions ===")
            print(f"Mean mAP: {mean_map:.2f}%")
            print(f"Mean R@1: {mean_r1:.2f}%")
            if not np.isnan(mean_r5):
                print(f"Mean R@5: {mean_r5:.2f}%")
            print("Per-lesion:")
            for row in summary_rows:
                print(
                    f"- {row['lesion']}: mAP {row['mAP']:.2f}% | "
                    f"R@1 {row['R@1']:.2f}% | "
                    f"fallback {row['fallback']} | reranked {row['reranked']}"
                )

        return 0
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
