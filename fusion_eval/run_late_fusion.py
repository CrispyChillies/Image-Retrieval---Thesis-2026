"""
Run late-fusion retrieval evaluation on stored ConvNeXtV2 and DINOv2 embeddings.

Required inputs:
- two embedding sources, one for ConvNeXtV2 and one for DINOv2
- each source must expose `image_path`, `label`, and `embedding`
- source type can be `milvus` or local `file` (`.json` or `.npz`)
- optional `query_set_path` to restrict evaluation to a specific ordered subset

How to run:
- create a JSON config file describing both sources and output settings
- run: `python -m fusion_eval.run_late_fusion --config path/to/config.json`

Produced outputs:
- console summary table for all baseline and fusion experiments
- `late_fusion_results.json`
- `late_fusion_results.csv`
- optional fused embedding `.npz` files if enabled in config

Score fusion:
- set `include_score_fusion` to true in config to run
  `score_fusion_alpha_X` experiments using
  `s = alpha * s_conv + (1 - alpha) * s_dino`
- optional `score_normalization`: `none`, `zscore`, or `minmax`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from fusion_eval.align import align_embedding_sources, build_embedding_source
from fusion_eval.evaluate import maybe_save_fused_embeddings, run_late_fusion_experiments
from fusion_eval.fuse import concat_fusion, weighted_sum_fusion
from retrieval_analysis.export_utils import ensure_dir, write_csv, write_json


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_results_table(rows: List[Dict[str, Any]]) -> str:
    headers = ["experiment", "samples", "mP@1", "mP@5", "mP@10", "R@1", "R@5", "R@10", "mAP", "status"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    def render(row: Dict[str, Any]) -> str:
        return " | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers)

    separator = "-+-".join("-" * widths[header] for header in headers)
    return "\n".join([render({header: header for header in headers}), separator] + [render(row) for row in rows])


def experiment_rows(experiments) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for experiment in experiments:
        if experiment.skipped:
            rows.append(
                {
                    "experiment": experiment.experiment_name,
                    "samples": experiment.num_samples,
                    "mP@1": "",
                    "mP@5": "",
                    "mP@10": "",
                    "R@1": "",
                    "R@5": "",
                    "R@10": "",
                    "mAP": "",
                    "status": experiment.skipped_reason or "skipped",
                }
            )
            continue
        rows.append(
            {
                "experiment": experiment.experiment_name,
                "samples": experiment.num_samples,
                "mP@1": f"{experiment.metrics.get('mP@1', 0.0):.2f}",
                "mP@5": f"{experiment.metrics.get('mP@5', 0.0):.2f}",
                "mP@10": f"{experiment.metrics.get('mP@10', 0.0):.2f}",
                "R@1": f"{experiment.metrics.get('R@1', 0.0):.2f}",
                "R@5": f"{experiment.metrics.get('R@5', 0.0):.2f}",
                "R@10": f"{experiment.metrics.get('R@10', 0.0):.2f}",
                "mAP": f"{experiment.metrics.get('mAP', 0.0):.2f}",
                "status": "ok",
            }
        )
    return rows


def maybe_export_fused_embeddings(config: Dict[str, Any], output_dir: Path, aligned) -> None:
    save_config = config.get("save_fused_embeddings", {})
    if not save_config.get("enabled", False):
        return

    concat_path = output_dir / save_config.get("concat_name", "concat_fusion_embeddings.npz")
    maybe_save_fused_embeddings(
        path=str(concat_path),
        image_paths=aligned.image_paths,
        labels=aligned.labels,
        embeddings=concat_fusion(aligned.conv_embeddings, aligned.dino_embeddings),
    )

    for alpha in config.get("alpha_values", [0.2, 0.4, 0.5, 0.6, 0.8]):
        weighted = weighted_sum_fusion(aligned.conv_embeddings, aligned.dino_embeddings, alpha)
        if weighted.embeddings is None:
            continue
        alpha_str = str(alpha).replace(".", "_")
        weighted_path = output_dir / f"weighted_sum_alpha_{alpha_str}.npz"
        maybe_save_fused_embeddings(
            path=str(weighted_path),
            image_paths=aligned.image_paths,
            labels=aligned.labels,
            embeddings=weighted.embeddings,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Late-fusion retrieval evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    conv_source = build_embedding_source(config["conv_source"])
    dino_source = build_embedding_source(config["dino_source"])

    aligned = align_embedding_sources(
        conv_source=conv_source,
        dino_source=dino_source,
        query_set_path=config.get("query_set_path"),
        strict_label_check=config.get("strict_label_check", True),
    )

    experiments = run_late_fusion_experiments(
        aligned=aligned,
        alpha_values=config.get("alpha_values", [0.2, 0.4, 0.5, 0.6, 0.8]),
        k_values=config.get("k_values", [1, 5, 10]),
        include_score_fusion=config.get("include_score_fusion", True),
        score_normalization=config.get("score_normalization", "none"),
    )

    rows = experiment_rows(experiments)
    print("Coverage:")
    print(f"  Present in ConvNeXt only: {len(aligned.coverage['present_in_conv_only'])}")
    print(f"  Present in DINO only: {len(aligned.coverage['present_in_dino_only'])}")
    print(f"  Present in both: {len(aligned.coverage['present_in_both'])}")
    print()
    print(format_results_table(rows))

    output_dir = ensure_dir(config.get("output_dir", "fusion_eval_output"))
    payload = {
        "coverage": aligned.coverage,
        "num_evaluated_samples": len(aligned.image_paths),
        "results": [
            {
                "experiment_name": experiment.experiment_name,
                "num_samples": experiment.num_samples,
                "metrics": experiment.metrics,
                "skipped": experiment.skipped,
                "skipped_reason": experiment.skipped_reason,
            }
            for experiment in experiments
        ],
    }
    write_json(output_dir / "late_fusion_results.json", payload)
    write_csv(output_dir / "late_fusion_results.csv", rows)
    maybe_export_fused_embeddings(config, output_dir, aligned)


if __name__ == "__main__":
    main()
