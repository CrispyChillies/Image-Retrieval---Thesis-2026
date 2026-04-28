"""
Run dual-cluster retrieval comparison for ConvNeXtV2 vs DINOv2.

Required config values:
- query_set_path: CSV/JSON/TXT file with ordered queries. CSV/JSON should expose
  `image_path` and optionally `label`.
- ConvNeXt cluster config: uri or host/port, auth if needed, collection_name.
- DINO cluster config: uri or host/port, auth if needed, collection_name.
- Optional: vector field name, db name, search params, output directory.

How to run:
- Update the example config in `main()` or pass a JSON config file with
  `--config path/to/config.json`.
- Then run: `python -m retrieval_analysis.run_analysis --config your_config.json`

Example config:
{
  "query_set_path": "queries.csv",
  "output_dir": "retrieval_analysis_output",
  "top_k": 5,
  "correctness_top_k": 1,
  "skip_missing_queries": true,
  "conv_search_params": {"metric_type": "COSINE", "params": {"nprobe": 10}},
  "dino_search_params": {"metric_type": "COSINE", "params": {"nprobe": 10}},
  "conv": {
    "name": "convnextv2",
    "uri": "https://cluster-a.example",
    "token": "token-a",
    "collection_name": "covid_image_retrieval_convnextv2"
  },
  "dino": {
    "name": "dinov2",
    "uri": "https://cluster-b.example",
    "token": "token-b",
    "collection_name": "covid_image_retrieval_dinov2"
  }
}

Generated files:
- `comparison_results.json`: full nested per-query analysis plus coverage/errors.
- `comparison_results.csv`: flattened per-query export.
- `group_<name>.csv`: one CSV per comparison group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from retrieval_analysis.comparison import ComparisonConfig, compare_models, load_query_set
from retrieval_analysis.evaluator import CorrectnessConfig
from retrieval_analysis.export_utils import ensure_dir, flatten_query_result, write_csv, write_json
from retrieval_analysis.milvus_adapter import MilvusCollectionAdapter, MilvusCollectionConfig


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_adapter(config: Dict[str, Any]) -> MilvusCollectionAdapter:
    return MilvusCollectionAdapter(MilvusCollectionConfig(**config))


def export_analysis(payload: Dict[str, Any], output_dir: str | Path) -> None:
    output_root = ensure_dir(output_dir)
    write_json(output_root / "comparison_results.json", payload)

    flat_rows = [flatten_query_result(row) for row in payload["results"]]
    write_csv(output_root / "comparison_results.csv", flat_rows)

    for group_name in (
        "both_correct",
        "both_wrong",
        "dino_correct_conv_wrong",
        "conv_correct_dino_wrong",
    ):
        group_rows = [
            flatten_query_result(row)
            for row in payload["results"]
            if row["assigned_group"] == group_name
        ]
        write_csv(output_root / f"group_{group_name}.csv", group_rows)


def print_summary(payload: Dict[str, Any]) -> None:
    summary = payload["summary"]
    coverage = payload["coverage"]
    print("Coverage:")
    print(f"  Present in ConvNeXt only: {len(coverage['present_in_conv_only'])}")
    print(f"  Present in DINO only: {len(coverage['present_in_dino_only'])}")
    print(f"  Present in both: {len(coverage['present_in_both'])}")
    print("Summary:")
    print(f"  both_correct: {summary['both_correct']}")
    print(f"  both_wrong: {summary['both_wrong']}")
    print(
        "  dino_correct_conv_wrong: "
        f"{summary['dino_correct_conv_wrong']}"
    )
    print(
        "  conv_correct_dino_wrong: "
        f"{summary['conv_correct_dino_wrong']}"
    )
    print(f"  evaluated_queries: {summary['evaluated_queries']}")
    print(f"  missing_queries: {len(payload['missing_queries'])}")
    print(f"  errors: {len(payload['errors'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-cluster retrieval comparison")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    raw_config = load_config(args.config)
    conv_adapter = build_adapter(raw_config["conv"])
    dino_adapter = build_adapter(raw_config["dino"])
    queries = load_query_set(raw_config["query_set_path"])

    comparison_config = ComparisonConfig(
        top_k=raw_config.get("top_k", 5),
        conv_search_params=raw_config.get("conv_search_params"),
        dino_search_params=raw_config.get("dino_search_params"),
        correctness=CorrectnessConfig(
            top_k=raw_config.get("correctness_top_k", 1)
        ),
        skip_missing_queries=raw_config.get("skip_missing_queries", True),
        preload_batch_size=raw_config.get("preload_batch_size", 100),
        search_batch_size=raw_config.get("search_batch_size", 50),
    )

    payload = compare_models(
        conv_adapter=conv_adapter,
        dino_adapter=dino_adapter,
        queries=queries,
        config=comparison_config,
    )
    export_analysis(payload, raw_config.get("output_dir", "retrieval_analysis_output"))
    print_summary(payload)


if __name__ == "__main__":
    main()
