import os
import glob
import yaml
from PIL import Image
import matplotlib.pyplot as plt

from .retrieval_system import ImageRetrievalSystem
from .segmenter import ImageSegmenter

INDEX_NAME = "image_index"


def _index_file(directory, version):
    """data/index/ + version 2 → data/index/image_index_v2.npz"""
    return os.path.join(directory, f"{INDEX_NAME}_v{version}.npz")


def _find_latest(directory):
    """Find the highest versioned index in a directory. Returns (path, version) or None."""
    pattern = os.path.join(directory, f"{INDEX_NAME}_v*.npz")
    files = glob.glob(pattern)

    if not files:
        return None

    best_version = 0
    best_path = None
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        try:
            v = int(basename.split("_v")[-1])
            if v > best_version:
                best_version = v
                best_path = f
        except ValueError:
            continue

    return (best_path, best_version) if best_path else None


def _show_results(query_path, matches, crops_dir):
    """Show query image alongside matched crops with details."""
    num_matches = len(matches)
    fig, axes = plt.subplots(1, num_matches + 1, figsize=(5 * (num_matches + 1), 5))

    # handle single match case (axes isn't a list)
    if num_matches == 1:
        axes = [axes[0], axes[1]]

    # query image
    query_img = Image.open(query_path)
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # matched crops
    for i, (label, distance, date_added) in enumerate(matches):
        crop_path = os.path.join(crops_dir, f"{label}.jpg")

        if os.path.exists(crop_path):
            crop_img = Image.open(crop_path)
        else:
            crop_img = Image.new("RGB", (224, 224), (200, 200, 200))

        axes[i + 1].imshow(crop_img)

        # show just the date part of the ISO timestamp
        date_short = date_added[:10] if len(date_added) >= 10 else date_added

        axes[i + 1].set_title(
            f"Match {i + 1}: {label}\n"
            f"Cosine Distance: {distance:.3f}\n"
            f"Added: {date_short}",
            fontsize=11,
        )
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    index_dir = cfg["index_dir"]
    crops_dir = os.path.join(index_dir, "crops")
    segmenter = ImageSegmenter()
    extensions = set(cfg.get("supported_extensions", [".png", ".jpg", ".jpeg", ".webp"]))

    if cfg["mode"] == "index":
        if "version" in cfg:
            version = cfg["version"]
        else:
            latest = _find_latest(index_dir)
            version = (latest[1] + 1) if latest else 1

        save_path = _index_file(index_dir, version)

        system = ImageRetrievalSystem(model=cfg.get("model", "vit_l_16"))
        system.index_images(cfg["database_dir"], segmenter=segmenter, crops_dir=crops_dir,
                            supported_extensions=extensions)
        system.save(save_path, version=version)

    elif cfg["mode"] == "search":
        if "version" in cfg:
            load_path = _index_file(index_dir, cfg["version"])
        else:
            latest = _find_latest(index_dir)
            if not latest:
                raise SystemExit(f"No index found in {index_dir}")
            load_path = latest[0]

        if not os.path.exists(load_path):
            raise SystemExit(f"Index not found: {load_path}")

        system = ImageRetrievalSystem(
            model=cfg.get("model", "vit_l_16"),
            index_path=load_path,
        )

        top_k = cfg.get("top_k", 5)
        threshold = cfg.get("threshold", 0.5)

        results = system.search(cfg["query_image"], k=top_k, segmenter=segmenter)

        # keep only results where cosine distance < threshold
        matches = []
        for label, distance, date_added in results:
            if distance <= threshold:
                matches.append((label, distance, date_added))

        if not matches:
            print("No matches found.")
            return

        print(f"\n{len(matches)} match(es) for: {cfg['query_image']}")
        for i, (label, distance, date_added) in enumerate(matches, 1):
            print(f"  {i}. {label}  (distance: {distance:.3f}, added: {date_added[:10]})")

        _show_results(cfg["query_image"], matches, crops_dir)

    else:
        raise SystemExit(f"Unknown mode: {cfg['mode']}. Use 'index' or 'search'.")


if __name__ == "__main__":
    main()
