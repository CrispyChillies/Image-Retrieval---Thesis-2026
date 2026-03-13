import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_QUERY_DIR = "/kaggle/input/isic-2017/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data"
DEFAULT_RETRIEVED_DIR = "/kaggle/input/isic-2017/ISIC-2017_Training_Data/ISIC-2017_Training_Data"
DEFAULT_OUTPUT_DIR = "/kaggle/working/saliency_overlays"
COMMON_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
	parser = argparse.ArgumentParser(
		description="Draw saliency overlays on retrieved images from evaluation JSON and saved .npy saliency maps."
	)
	parser.add_argument("--results-json", type=str, required=True,
						help="Path to the evaluation JSON file.")
	parser.add_argument("--query-id", type=str, required=True,
						help="Query identifier, image id, or filename to visualize.")
	parser.add_argument("--saliency-dir", type=str, default=None,
						help="Directory that already contains rank*.npy files for one query.")
	parser.add_argument("--saliency-root", type=str, default=None,
						help="Root directory that contains per-query saliency folders.")
	parser.add_argument("--query-dir", type=str, default=DEFAULT_QUERY_DIR,
						help="Directory containing query images.")
	parser.add_argument("--retrieved-dir", type=str, default=DEFAULT_RETRIEVED_DIR,
						help="Directory containing retrieved images. Overlays are drawn on these images.")
	parser.add_argument("--retrieved-images", nargs="*", default=None,
						help="Optional override list of retrieved image filenames, ordered by rank.")
	parser.add_argument("--top-k", type=int, default=None,
						help="Optional limit on how many ranks to render.")
	parser.add_argument("--alpha", type=float, default=0.45,
						help="Overlay strength in [0, 1].")
	parser.add_argument("--cmap", type=str, default="jet",
						help="Matplotlib colormap name for the heatmap.")
	parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
						help="Directory to save output images.")
	parser.add_argument("--save-grid", action="store_true",
						help="Also save a single multi-row grid figure for all ranks.")
	return parser.parse_args()


def normalize_text(value):
	if value is None:
		return ""
	text = str(value).strip()
	return text.lower()


def query_keys(query_result):
	keys = set()
	query_image = query_result.get("query_image", "")
	query_image_id = query_result.get("query_image_id", "")

	for candidate in (query_image, query_image_id):
		if not candidate:
			continue
		candidate_path = Path(str(candidate))
		keys.add(normalize_text(candidate))
		keys.add(normalize_text(candidate_path.name))
		keys.add(normalize_text(candidate_path.stem))

	return keys


def load_query_result(results_json_path, query_id):
	with open(results_json_path, "r", encoding="utf-8") as handle:
		data = json.load(handle)

	results = data.get("results", [])
	query_key = normalize_text(query_id)

	for query_result in results:
		if query_key in query_keys(query_result):
			return query_result

	raise ValueError(f"Could not find query '{query_id}' in {results_json_path}.")


def candidate_names_from_value(value):
	if not value:
		return []

	path = Path(str(value))
	stem = path.stem
	names = [path.name]
	if stem and stem != path.name:
		names.extend([stem + ext for ext in COMMON_EXTENSIONS])
	if path.suffix:
		names.append(stem)
	return list(dict.fromkeys(names))


def resolve_image_path(image_dir, candidates):
	image_dir = Path(image_dir)
	for candidate in candidates:
		candidate_path = image_dir / candidate
		if candidate_path.exists():
			return candidate_path

	lower_map = {child.name.lower(): child for child in image_dir.iterdir() if child.is_file()}
	for candidate in candidates:
		found = lower_map.get(candidate.lower())
		if found is not None:
			return found

	raise FileNotFoundError(
		f"Could not resolve image in {image_dir} from candidates: {candidates}"
	)


def resolve_query_image(query_result, query_dir):
	candidates = []
	candidates.extend(candidate_names_from_value(query_result.get("query_image")))
	candidates.extend(candidate_names_from_value(query_result.get("query_image_id")))
	return resolve_image_path(query_dir, list(dict.fromkeys(candidates)))


def infer_saliency_dir(query_result, saliency_root):
	query_image = query_result.get("query_image", "")
	query_id = query_result.get("query_image_id", "")
	candidate_dirs = []

	for value in (query_image, query_id):
		if not value:
			continue
		value_path = Path(str(value))
		candidate_dirs.append(value_path.name.replace(".", "_"))
		candidate_dirs.append(value_path.stem.replace(".", "_"))
		candidate_dirs.append(str(value).replace(".", "_"))

	saliency_root = Path(saliency_root)
	for candidate in dict.fromkeys(candidate_dirs):
		path = saliency_root / candidate
		if path.exists() and path.is_dir():
			return path

	raise FileNotFoundError(
		f"Could not infer saliency directory under {saliency_root} for query {query_result.get('query_image')}"
	)


def extract_rank(path):
	digits = "".join(char for char in path.stem if char.isdigit())
	if not digits:
		raise ValueError(f"Could not extract rank from file name: {path.name}")
	return int(digits)


def load_saliency_files(saliency_dir, top_k=None):
	saliency_dir = Path(saliency_dir)
	files = sorted(saliency_dir.glob("rank*_saliency.npy"), key=extract_rank)
	if top_k is not None:
		files = files[:top_k]
	if not files:
		raise FileNotFoundError(f"No rank*_saliency.npy files found in {saliency_dir}")
	return files


def resolve_retrieved_images(query_result, retrieved_dir, saliency_files, retrieved_override=None):
	if retrieved_override:
		if len(retrieved_override) < len(saliency_files):
			raise ValueError(
				"The number of --retrieved-images must be at least the number of saliency files."
			)
		resolved = []
		for image_name in retrieved_override[:len(saliency_files)]:
			resolved.append(resolve_image_path(retrieved_dir, candidate_names_from_value(image_name)))
		return resolved

	retrieved_entries = query_result.get("retrieved", [])
	if not retrieved_entries:
		raise ValueError(
			"This JSON entry does not contain per-rank retrieved images. "
			"Pass --retrieved-images explicitly, or use a detailed results JSON with query_result['retrieved']."
		)

	rank_to_entry = {}
	for entry in retrieved_entries:
		rank = entry.get("rank")
		if rank is not None:
			rank_to_entry[int(rank)] = entry

	resolved = []
	for saliency_file in saliency_files:
		rank = extract_rank(saliency_file)
		entry = rank_to_entry.get(rank)
		if entry is None:
			raise ValueError(f"Missing retrieved metadata for rank {rank}")
		candidates = candidate_names_from_value(entry.get("retrieved_image"))
		resolved.append(resolve_image_path(retrieved_dir, candidates))
	return resolved


def normalize_saliency(saliency):
	saliency = np.asarray(saliency, dtype=np.float32)
	saliency = np.squeeze(saliency)
	if saliency.ndim != 2:
		raise ValueError(f"Expected a 2D saliency map, got shape {saliency.shape}")
	min_value = float(np.min(saliency))
	max_value = float(np.max(saliency))
	if max_value <= min_value:
		return np.zeros_like(saliency, dtype=np.float32)
	return np.interp(saliency, (min_value, max_value), (0.0, 1.0)).astype(np.float32)


def resize_saliency(saliency, size):
	saliency_uint8 = np.clip(saliency * 255.0, 0, 255).astype(np.uint8)
	resized = Image.fromarray(saliency_uint8).resize(size, Image.Resampling.BILINEAR)
	return np.asarray(resized, dtype=np.float32) / 255.0


def make_overlay(image, saliency, alpha=0.45, cmap_name="jet"):
	image_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
	resized_saliency = resize_saliency(saliency, image.size)
	colored = plt.get_cmap(cmap_name)(resized_saliency)[..., :3]
	overlay = (1.0 - alpha) * image_rgb + alpha * colored
	overlay = np.clip(overlay, 0.0, 1.0)
	return overlay, resized_saliency


def save_rank_figure(output_path, query_image, retrieved_image, overlay, rank, similarity=None):
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	axes[0].imshow(query_image)
	axes[0].set_title("Query")
	axes[0].axis("off")

	axes[1].imshow(retrieved_image)
	if similarity is None:
		axes[1].set_title(f"Retrieved rank {rank}")
	else:
		axes[1].set_title(f"Retrieved rank {rank} | sim={similarity:.4f}")
	axes[1].axis("off")

	axes[2].imshow(overlay)
	axes[2].set_title(f"Overlay rank {rank}")
	axes[2].axis("off")

	fig.tight_layout()
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def save_grid_figure(output_path, query_image, rows):
	row_count = len(rows)
	fig, axes = plt.subplots(row_count, 3, figsize=(15, 5 * row_count))
	if row_count == 1:
		axes = np.expand_dims(axes, axis=0)

	for index, row in enumerate(rows):
		axes[index, 0].imshow(query_image)
		axes[index, 0].set_title("Query" if index == 0 else "")
		axes[index, 0].axis("off")

		axes[index, 1].imshow(row["retrieved_image"])
		similarity = row.get("similarity")
		if similarity is None:
			title = f"Retrieved rank {row['rank']}"
		else:
			title = f"Retrieved rank {row['rank']} | sim={similarity:.4f}"
		axes[index, 1].set_title(title)
		axes[index, 1].axis("off")

		axes[index, 2].imshow(row["overlay"])
		axes[index, 2].set_title(f"Overlay rank {row['rank']}")
		axes[index, 2].axis("off")

	fig.tight_layout()
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def main():
	args = parse_args()
	if not args.saliency_dir and not args.saliency_root:
		raise ValueError("Provide either --saliency-dir or --saliency-root.")

	query_result = load_query_result(args.results_json, args.query_id)
	saliency_dir = Path(args.saliency_dir) if args.saliency_dir else infer_saliency_dir(query_result, args.saliency_root)
	saliency_files = load_saliency_files(saliency_dir, top_k=args.top_k)

	query_image_path = resolve_query_image(query_result, args.query_dir)
	query_image = Image.open(query_image_path).convert("RGB")

	retrieved_paths = resolve_retrieved_images(
		query_result=query_result,
		retrieved_dir=args.retrieved_dir,
		saliency_files=saliency_files,
		retrieved_override=args.retrieved_images,
	)

	output_dir = Path(args.output_dir) / normalize_text(query_result.get("query_image_id") or query_result.get("query_image") or args.query_id)
	output_dir.mkdir(parents=True, exist_ok=True)

	similarity_by_rank = {
		int(item["rank"]): item.get("similarity")
		for item in query_result.get("retrieved", [])
		if item.get("rank") is not None
	}

	grid_rows = []
	for saliency_file, retrieved_path in zip(saliency_files, retrieved_paths):
		rank = extract_rank(saliency_file)
		saliency = normalize_saliency(np.load(saliency_file))
		retrieved_image = Image.open(retrieved_path).convert("RGB")
		overlay, _ = make_overlay(retrieved_image, saliency, alpha=args.alpha, cmap_name=args.cmap)

		rank_output = output_dir / f"rank{rank}_overlay.png"
		save_rank_figure(
			output_path=rank_output,
			query_image=query_image,
			retrieved_image=retrieved_image,
			overlay=overlay,
			rank=rank,
			similarity=similarity_by_rank.get(rank),
		)

		grid_rows.append({
			"rank": rank,
			"retrieved_image": retrieved_image,
			"overlay": overlay,
			"similarity": similarity_by_rank.get(rank),
		})

		print(f"Saved rank {rank} overlay to {rank_output}")

	if args.save_grid:
		grid_output = output_dir / "all_ranks_overlay_grid.png"
		save_grid_figure(grid_output, query_image, grid_rows)
		print(f"Saved grid figure to {grid_output}")


if __name__ == "__main__":
	main()
