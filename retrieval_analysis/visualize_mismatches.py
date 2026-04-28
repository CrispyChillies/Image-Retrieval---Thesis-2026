"""Generate an HTML gallery for retrieval mismatch cases."""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_GROUPS = ("dino_correct_conv_wrong", "both_wrong")


def load_results(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_groups(groups: Sequence[str] | None) -> List[str]:
    if not groups:
        return list(DEFAULT_GROUPS)
    normalized: List[str] = []
    for group in groups:
        for token in group.split(","):
            stripped = token.strip()
            if stripped:
                normalized.append(stripped)
    return normalized or list(DEFAULT_GROUPS)


def filter_results(
    payload: Dict,
    groups: Sequence[str],
    max_per_group: Optional[int] = None,
) -> List[Dict]:
    selected: List[Dict] = []
    for group in groups:
        group_rows = [
            row for row in payload.get("results", [])
            if row.get("assigned_group") == group
        ]
        if max_per_group is not None:
            group_rows = group_rows[:max_per_group]
        selected.extend(group_rows)
    return selected


def resolve_image_path(
    raw_path: str,
    path_mappings: Sequence[Tuple[str, str]],
) -> Path:
    for source_prefix, target_prefix in path_mappings:
        if raw_path.startswith(source_prefix):
            remapped = target_prefix + raw_path[len(source_prefix):]
            return Path(remapped)
    return Path(raw_path)


def image_data_uri(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def render_image_tile(
    title: str,
    raw_path: str,
    label: Optional[str],
    score: Optional[float],
    path_mappings: Sequence[Tuple[str, str]],
) -> str:
    resolved_path = resolve_image_path(raw_path, path_mappings)
    data_uri = image_data_uri(resolved_path)
    safe_title = html.escape(title)
    safe_raw_path = html.escape(raw_path)
    safe_resolved_path = html.escape(str(resolved_path))
    safe_label = html.escape(label or "unknown")
    score_text = "" if score is None else f"{score:.4f}"

    if data_uri is not None:
        media_html = (
            f'<img src="{data_uri}" alt="{safe_title}" loading="lazy" />'
        )
    else:
        media_html = (
            '<div class="missing-image">'
            "Image not found locally"
            "</div>"
        )

    return f"""
    <div class="image-tile">
      <div class="tile-title">{safe_title}</div>
      <div class="tile-media">{media_html}</div>
      <div class="tile-meta"><strong>Label:</strong> {safe_label}</div>
      <div class="tile-meta"><strong>Score:</strong> {html.escape(score_text)}</div>
      <div class="tile-path">{safe_raw_path}</div>
      <div class="tile-path resolved">{safe_resolved_path}</div>
    </div>
    """


def render_hits_column(
    model_name: str,
    result: Dict,
    correct: bool,
    path_mappings: Sequence[Tuple[str, str]],
    top_k: int,
) -> str:
    hits = result.get("hits", [])[:top_k]
    tiles = [
        render_image_tile(
            title=f"{model_name} rank {index + 1}",
            raw_path=hit.get("image_path", ""),
            label=hit.get("label"),
            score=hit.get("score"),
            path_mappings=path_mappings,
        )
        for index, hit in enumerate(hits)
    ]
    status_class = "correct" if correct else "wrong"
    status_label = "Correct" if correct else "Wrong"
    return f"""
    <section class="result-column {status_class}">
      <h3>{html.escape(model_name)} <span>{status_label}</span></h3>
      <div class="tiles">
        {''.join(tiles)}
      </div>
    </section>
    """


def render_case_card(
    row: Dict,
    path_mappings: Sequence[Tuple[str, str]],
    top_k: int,
) -> str:
    query_tile = render_image_tile(
        title="Query",
        raw_path=row.get("query_image_path", ""),
        label=row.get("query_label"),
        score=None,
        path_mappings=path_mappings,
    )
    group_name = html.escape(row.get("assigned_group", "unknown"))
    return f"""
    <article class="case-card">
      <header class="case-header">
        <div>
          <div class="case-group">{group_name}</div>
          <div class="case-label">Query label: {html.escape(str(row.get("query_label", "unknown")))}</div>
        </div>
      </header>
      <div class="case-grid">
        <section class="query-column">
          <h3>Query</h3>
          <div class="tiles">{query_tile}</div>
        </section>
        {render_hits_column("ConvNeXt", row.get("conv", {}), bool(row.get("conv_correct")), path_mappings, top_k)}
        {render_hits_column("DINO", row.get("dino", {}), bool(row.get("dino_correct")), path_mappings, top_k)}
      </div>
    </article>
    """


def render_html(
    rows: Sequence[Dict],
    groups: Sequence[str],
    top_k: int,
    path_mappings: Sequence[Tuple[str, str]],
) -> str:
    cards = "".join(
        render_case_card(row=row, path_mappings=path_mappings, top_k=top_k)
        for row in rows
    )
    group_list = ", ".join(groups)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Retrieval Mismatch Gallery</title>
  <style>
    :root {{
      --bg: #f2efe8;
      --surface: #fffdf8;
      --ink: #1f2933;
      --muted: #67717d;
      --line: #d6d0c4;
      --wrong: #b5432a;
      --correct: #1f7a46;
      --accent: #c08a28;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff7dd 0, transparent 28%),
        linear-gradient(180deg, #f5f2ea 0%, #ece7dc 100%);
    }}
    .page {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    .hero {{
      margin-bottom: 28px;
      padding: 24px 28px;
      border: 1px solid var(--line);
      background: rgba(255, 253, 248, 0.82);
      backdrop-filter: blur(8px);
    }}
    h1, h2, h3 {{ margin: 0; font-weight: 600; }}
    .subtitle {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 16px;
    }}
    .case-card {{
      margin-top: 24px;
      border: 1px solid var(--line);
      background: var(--surface);
      box-shadow: 0 20px 40px rgba(68, 52, 24, 0.08);
    }}
    .case-header {{
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(192, 138, 40, 0.1), rgba(255, 255, 255, 0));
    }}
    .case-group {{
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
    }}
    .case-label {{
      margin-top: 6px;
      font-size: 18px;
    }}
    .case-grid {{
      display: grid;
      grid-template-columns: 1fr 1.5fr 1.5fr;
      gap: 16px;
      padding: 16px;
    }}
    .query-column, .result-column {{
      border: 1px solid var(--line);
      padding: 12px;
      background: #fff;
    }}
    .result-column.correct h3 span {{ color: var(--correct); }}
    .result-column.wrong h3 span {{ color: var(--wrong); }}
    .tiles {{
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }}
    .image-tile {{
      border: 1px solid var(--line);
      padding: 10px;
      background: #fcfaf5;
    }}
    .tile-title {{
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .tile-media {{
      aspect-ratio: 1 / 1;
      background: #ebe5d8;
      display: grid;
      place-items: center;
      overflow: hidden;
    }}
    .tile-media img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #fff;
    }}
    .missing-image {{
      color: var(--muted);
      font-size: 13px;
      padding: 16px;
      text-align: center;
    }}
    .tile-meta {{
      margin-top: 8px;
      font-size: 13px;
    }}
    .tile-path {{
      margin-top: 8px;
      font-size: 11px;
      color: var(--muted);
      word-break: break-all;
    }}
    .resolved {{
      font-style: italic;
    }}
    @media (max-width: 1100px) {{
      .case-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>Retrieval Mismatch Gallery</h1>
      <div class="subtitle">Groups: {html.escape(group_list)} | Cases: {len(rows)} | Top hits shown per model: {top_k}</div>
    </section>
    {cards}
  </main>
</body>
</html>
"""


def parse_path_mappings(values: Sequence[str]) -> List[Tuple[str, str]]:
    mappings: List[Tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --path-map value {value!r}. Expected format: source_prefix=target_prefix"
            )
        source_prefix, target_prefix = value.split("=", 1)
        mappings.append((source_prefix, target_prefix))
    return mappings


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize retrieval mismatch cases as HTML")
    parser.add_argument(
        "--input",
        type=str,
        default="retrieval_analysis_output/comparison_results.json",
        help="Path to comparison_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="retrieval_analysis_output/mismatch_gallery.html",
        help="Path to generated HTML file",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="*",
        default=list(DEFAULT_GROUPS),
        help="Mismatch groups to include",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved images to show per model",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=None,
        help="Optional cap on number of cases rendered per group",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Optional path prefix remap in the form source_prefix=target_prefix",
    )
    args = parser.parse_args()

    payload = load_results(args.input)
    groups = normalize_groups(args.groups)
    rows = filter_results(payload, groups, max_per_group=args.max_per_group)
    path_mappings = parse_path_mappings(args.path_map)
    html_text = render_html(
        rows=rows,
        groups=groups,
        top_k=args.top_k,
        path_mappings=path_mappings,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote mismatch gallery to {output_path}")


if __name__ == "__main__":
    main()
