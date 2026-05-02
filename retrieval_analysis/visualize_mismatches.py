"""Generate HTML galleries and annotation scaffolds for retrieval mismatch cases."""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_GROUPS = ("dino_correct_conv_wrong", "both_wrong")
DEFAULT_PATTERN_TAGS = (
    "bilateral_opacity",
    "low_contrast_lungs",
    "rotated_ap_portable_view",
    "severe_pneumonia_confused_with_covid19",
)


def load_results(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_annotations(path: str | Path) -> Dict[str, Dict]:
    annotation_path = Path(path)
    if not annotation_path.exists():
        return {}
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("annotations", payload)
    return {
        str(item["query_image_path"]): item
        for item in entries
        if item.get("query_image_path")
    }


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
        media_html = f'<img src="{data_uri}" alt="{safe_title}" loading="lazy" />'
    else:
        media_html = '<div class="missing-image">Image not found locally</div>'

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
      <div class="tiles">{''.join(tiles)}</div>
    </section>
    """


def normalize_tags(tags: Sequence[str]) -> List[str]:
    return [tag.strip() for tag in tags if tag and tag.strip()]


def render_annotation_summary(annotation: Optional[Dict]) -> str:
    if not annotation:
        return '<div class="annotation-box"><strong>Pattern tags:</strong> not annotated</div>'
    tags = normalize_tags(annotation.get("pattern_tags", []))
    notes = html.escape(annotation.get("notes", ""))
    tag_html = ", ".join(html.escape(tag) for tag in tags) or "none"
    note_html = notes or "none"
    return (
        '<div class="annotation-box">'
        f"<div><strong>Pattern tags:</strong> {tag_html}</div>"
        f"<div><strong>Notes:</strong> {note_html}</div>"
        "</div>"
    )


def render_case_card(
    row: Dict,
    path_mappings: Sequence[Tuple[str, str]],
    top_k: int,
    annotation: Optional[Dict] = None,
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
      <div class="case-summary">
        {render_annotation_summary(annotation)}
      </div>
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


def page_style() -> str:
    return """
  <style>
    :root {
      --bg: #f2efe8;
      --surface: #fffdf8;
      --ink: #1f2933;
      --muted: #67717d;
      --line: #d6d0c4;
      --wrong: #b5432a;
      --correct: #1f7a46;
      --accent: #c08a28;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff7dd 0, transparent 28%),
        linear-gradient(180deg, #f5f2ea 0%, #ece7dc 100%);
    }
    .page {
      max-width: 1600px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }
    .hero {
      margin-bottom: 28px;
      padding: 24px 28px;
      border: 1px solid var(--line);
      background: rgba(255, 253, 248, 0.82);
      backdrop-filter: blur(8px);
    }
    h1, h2, h3 { margin: 0; font-weight: 600; }
    .subtitle {
      margin-top: 8px;
      color: var(--muted);
      font-size: 16px;
    }
    .section-title {
      margin-top: 32px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
    }
    .case-card {
      margin-top: 24px;
      border: 1px solid var(--line);
      background: var(--surface);
      box-shadow: 0 20px 40px rgba(68, 52, 24, 0.08);
    }
    .case-header {
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(192, 138, 40, 0.1), rgba(255, 255, 255, 0));
    }
    .case-summary {
      padding: 12px 16px 0;
    }
    .case-group {
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .case-label {
      margin-top: 6px;
      font-size: 18px;
    }
    .case-grid {
      display: grid;
      grid-template-columns: 1fr 1.5fr 1.5fr;
      gap: 16px;
      padding: 16px;
    }
    .query-column, .result-column {
      border: 1px solid var(--line);
      padding: 12px;
      background: #fff;
    }
    .result-column.correct h3 span { color: var(--correct); }
    .result-column.wrong h3 span { color: var(--wrong); }
    .tiles {
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }
    .image-tile {
      border: 1px solid var(--line);
      padding: 10px;
      background: #fcfaf5;
    }
    .tile-title {
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .tile-media {
      aspect-ratio: 1 / 1;
      background: #ebe5d8;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    .tile-media img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #fff;
    }
    .missing-image {
      color: var(--muted);
      font-size: 13px;
      padding: 16px;
      text-align: center;
    }
    .tile-meta {
      margin-top: 8px;
      font-size: 13px;
    }
    .tile-path {
      margin-top: 8px;
      font-size: 11px;
      color: var(--muted);
      word-break: break-all;
    }
    .resolved { font-style: italic; }
    .annotation-box {
      border: 1px dashed var(--line);
      padding: 10px 12px;
      background: #fff;
      font-size: 13px;
    }
    .pattern-block {
      margin-top: 32px;
    }
    .pattern-count {
      color: var(--muted);
      font-size: 14px;
      margin-top: 6px;
    }
    .annotation-item {
      border: 1px solid var(--line);
      background: #fff;
      padding: 16px;
      margin-top: 16px;
    }
    .annotation-item textarea {
      width: 100%;
      min-height: 72px;
      margin-top: 8px;
      padding: 8px;
      border: 1px solid var(--line);
      font: inherit;
    }
    .tag-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .tag-pill {
      display: inline-block;
      padding: 2px 8px;
      margin-right: 8px;
      border: 1px solid var(--line);
      background: #fff;
      font-size: 12px;
    }
    code {
      background: rgba(0, 0, 0, 0.04);
      padding: 0 4px;
    }
    @media (max-width: 1100px) {
      .case-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
"""


def wrap_html(title: str, subtitle: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
{page_style()}
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <div class="subtitle">{html.escape(subtitle)}</div>
    </section>
    {body}
  </main>
</body>
</html>
"""


def render_gallery_html(
    rows: Sequence[Dict],
    groups: Sequence[str],
    top_k: int,
    path_mappings: Sequence[Tuple[str, str]],
    annotations: Dict[str, Dict],
) -> str:
    cards = "".join(
        render_case_card(
            row=row,
            path_mappings=path_mappings,
            top_k=top_k,
            annotation=annotations.get(str(row.get("query_image_path", ""))),
        )
        for row in rows
    )
    subtitle = (
        f"Groups: {', '.join(groups)} | Cases: {len(rows)} | "
        f"Top hits shown per model: {top_k}"
    )
    return wrap_html("Retrieval Mismatch Gallery", subtitle, cards)


def build_annotation_payload(
    rows: Sequence[Dict],
    pattern_tags: Sequence[str],
    existing_annotations: Dict[str, Dict],
) -> Dict:
    annotations: List[Dict] = []
    for row in rows:
        query_image_path = str(row.get("query_image_path", ""))
        existing = existing_annotations.get(query_image_path, {})
        annotations.append(
            {
                "query_image_path": query_image_path,
                "query_label": row.get("query_label"),
                "assigned_group": row.get("assigned_group"),
                "pattern_tags": normalize_tags(existing.get("pattern_tags", [])),
                "notes": existing.get("notes", ""),
                "candidate_pattern_tags": list(pattern_tags),
            }
        )
    return {"annotations": annotations}


def render_annotation_form(
    rows: Sequence[Dict],
    pattern_tags: Sequence[str],
) -> str:
    blocks: List[str] = []
    for row in rows:
        query_image_path = str(row.get("query_image_path", ""))
        tags_markup = "".join(
            f'<label><input type="checkbox" disabled /> {html.escape(tag)}</label>'
            for tag in pattern_tags
        )
        blocks.append(
            f"""
            <section class="annotation-item">
              <div><strong>query_image_path</strong>: <code>{html.escape(query_image_path)}</code></div>
              <div><strong>group</strong>: {html.escape(str(row.get("assigned_group", "")))}</div>
              <div><strong>query_label</strong>: {html.escape(str(row.get("query_label", "")))}</div>
              <div class="tag-list">{tags_markup}</div>
              <textarea disabled>Use the JSON scaffold to fill pattern_tags and notes for this case.</textarea>
            </section>
            """
        )
    help_text = (
        "Review the main gallery, then edit the JSON scaffold and assign one or more "
        "pattern tags per case. Suggested starting tags are shown below."
    )
    body = (
        f'<section><p>{html.escape(help_text)}</p></section>'
        + "".join(blocks)
    )
    return wrap_html("Mismatch Annotation Guide", f"Cases: {len(rows)}", body)


def render_pattern_summary_html(
    rows: Sequence[Dict],
    top_k: int,
    path_mappings: Sequence[Tuple[str, str]],
    annotations: Dict[str, Dict],
) -> str:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    untagged: List[Dict] = []

    for row in rows:
        query_image_path = str(row.get("query_image_path", ""))
        annotation = annotations.get(query_image_path)
        tags = normalize_tags(annotation.get("pattern_tags", [])) if annotation else []
        if not tags:
            untagged.append(row)
            continue
        for tag in tags:
            grouped[tag].append(row)

    sections: List[str] = []
    for tag in sorted(grouped):
        cards = "".join(
            render_case_card(
                row=row,
                path_mappings=path_mappings,
                top_k=top_k,
                annotation=annotations.get(str(row.get("query_image_path", ""))),
            )
            for row in grouped[tag]
        )
        sections.append(
            f"""
            <section class="pattern-block">
              <h2 class="section-title">{html.escape(tag)}</h2>
              <div class="pattern-count">{len(grouped[tag])} cases</div>
              {cards}
            </section>
            """
        )

    if untagged:
        cards = "".join(
            render_case_card(
                row=row,
                path_mappings=path_mappings,
                top_k=top_k,
                annotation=annotations.get(str(row.get("query_image_path", ""))),
            )
            for row in untagged
        )
        sections.append(
            f"""
            <section class="pattern-block">
              <h2 class="section-title">untagged</h2>
              <div class="pattern-count">{len(untagged)} cases</div>
              {cards}
            </section>
            """
        )

    if not sections:
        sections.append(
            "<section><p>No annotated pattern tags found. Fill the annotation JSON scaffold first.</p></section>"
        )

    return wrap_html(
        "Mismatch Patterns",
        f"Annotated cases grouped by visual pattern | Top hits shown per model: {top_k}",
        "".join(sections),
    )


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


def parse_pattern_tags(values: Sequence[str]) -> List[str]:
    if not values:
        return list(DEFAULT_PATTERN_TAGS)
    normalized: List[str] = []
    for value in values:
        for token in value.split(","):
            stripped = token.strip()
            if stripped:
                normalized.append(stripped)
    return normalized or list(DEFAULT_PATTERN_TAGS)


def write_text(path: str | Path, content: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_json(path: str | Path, payload: Dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize retrieval mismatch cases and build pattern-grouped reports"
    )
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
        help="Path to generated mismatch gallery HTML",
    )
    parser.add_argument(
        "--grouped-output",
        type=str,
        default="retrieval_analysis_output/mismatch_patterns.html",
        help="Path to generated pattern-grouped HTML",
    )
    parser.add_argument(
        "--annotation-output",
        type=str,
        default="retrieval_analysis_output/mismatch_annotations.json",
        help="Path to annotation JSON scaffold",
    )
    parser.add_argument(
        "--annotation-guide-output",
        type=str,
        default="retrieval_analysis_output/mismatch_annotation_guide.html",
        help="Path to annotation guide HTML",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="*",
        default=list(DEFAULT_GROUPS),
        help="Mismatch groups to include",
    )
    parser.add_argument(
        "--pattern-tags",
        type=str,
        nargs="*",
        default=list(DEFAULT_PATTERN_TAGS),
        help="Suggested pattern tags for annotation scaffolding",
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
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Optional existing annotation JSON to use for grouped output",
    )
    args = parser.parse_args()

    payload = load_results(args.input)
    groups = normalize_groups(args.groups)
    pattern_tags = parse_pattern_tags(args.pattern_tags)
    rows = filter_results(payload, groups, max_per_group=args.max_per_group)
    path_mappings = parse_path_mappings(args.path_map)

    existing_annotations = (
        load_annotations(args.annotations) if args.annotations else {}
    )
    annotation_payload = build_annotation_payload(rows, pattern_tags, existing_annotations)
    annotation_path = write_json(args.annotation_output, annotation_payload)

    gallery_html = render_gallery_html(
        rows=rows,
        groups=groups,
        top_k=args.top_k,
        path_mappings=path_mappings,
        annotations=existing_annotations,
    )
    gallery_path = write_text(args.output, gallery_html)

    guide_html = render_annotation_form(rows, pattern_tags)
    guide_path = write_text(args.annotation_guide_output, guide_html)

    merged_annotations = load_annotations(annotation_path)
    if args.annotations:
        merged_annotations = load_annotations(args.annotations)

    grouped_html = render_pattern_summary_html(
        rows=rows,
        top_k=args.top_k,
        path_mappings=path_mappings,
        annotations=merged_annotations,
    )
    grouped_path = write_text(args.grouped_output, grouped_html)

    print(f"Wrote mismatch gallery to {gallery_path}")
    print(f"Wrote annotation scaffold to {annotation_path}")
    print(f"Wrote annotation guide to {guide_path}")
    print(f"Wrote pattern-grouped gallery to {grouped_path}")


if __name__ == "__main__":
    main()
