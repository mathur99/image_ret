# Image Retrieval (Lost & Found)

![Rapido Image Retrieval](readme_files/header.webp)

Image Retrieval is a Python-based tool for indexing and searching images using deep feature extraction and similarity search. It converts images to JPG, extracts Vision Transformer (ViT) embeddings, builds a FAISS index, and retrieves the most similar images given a query image.

## Features

![Lost & Found](readme_files/Figure_1.png)

- Convert various image formats to JPG for consistency (PNG, JPEG, WebP, etc.)
- Index images into a searchable feature database using ViT-L/16
- Search for similar images by content using cosine similarity
- Display similarity scores and matched images visually
- Optional GPU acceleration for both feature extraction and FAISS search

---

## Architecture

### Overview

The system has two main pipelines: **indexing** (build the searchable database) and **search** (query and retrieve similar images). Both rely on L2-normalized ViT embeddings and a FAISS inner-product index for fast similarity search.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INDEX PIPELINE                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Image directory  →  Convert to JPG  →  ViT feature extraction  →  FAISS index   │
│       (raw)           (utils)              (feature_extractor)      + metadata   │
│                                                                         ↓         │
│                                                    image_index.faiss + image_metadata.json
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SEARCH PIPELINE                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Query image  →  ViT feature extraction  →  FAISS k-NN (inner product)  →  Top-K │
│                      (same model)              cosine similarity            results │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Location | Role |
|-----------|----------|------|
| **Entry point** | `src/index_and_retrieve.py` | Orchestrates index vs search; calls conversion, indexing, and result printing. |
| **Image conversion** | `utils/convert_images_to_jpg.py` | Converts supported formats (from `utils/supported_image_types.json`) to JPG under `converted_jpgs/`. |
| **Feature extraction** | `src/feature_extractor.py` | Loads pretrained **ViT-L/16** (ImageNet1K), strips classifier head, outputs L2-normalized 1024-d vectors per image. |
| **Retrieval system** | `src/retrieval_system.py` | Builds/loads a **FAISS IndexFlatIP** (inner product = cosine for unit vectors); stores index ↔ path mapping in JSON metadata. |
| **Config** | `utils/supported_image_types.json` | Which file extensions to convert (e.g. `.png`, `.jpg`, `.webp`). |

### Data flow

1. **Index**
   - Input: directory of images (e.g. `support_database_images/`).
   - Images matching `supported_image_types.json` are converted to JPG in `image_dir/converted_jpgs/`.
   - Each JPG is passed through ViT-L/16; features are L2-normalized and added to the FAISS index.
   - Index IDs are mapped to image paths in `image_metadata.json`.
   - Output: `image_index.faiss` (FAISS index) and `image_metadata.json` (id → path).

2. **Search**
   - Input: path to a query image and optional `k`.
   - Query image is passed through the same ViT; its L2-normalized vector is used for FAISS search.
   - FAISS returns top-k indices and inner-product (cosine) scores.
   - Metadata maps indices to paths; results are sorted by similarity and returned (path, similarity, distance).
   - Optional: matplotlib shows query image and closest match.

### Why ViT + FAISS?

- **ViT-L/16**: Strong semantic image representation; 1024-d vectors work well for similarity without training.
- **L2 normalization**: Makes inner product equal to cosine similarity; FAISS `IndexFlatIP` is exact and simple.
- **FAISS**: Fast k-NN at scale; optional GPU for larger indexes.

### Project structure

```
image_ret/
├── README.md
├── pyproject.toml              # Package config, Hatch env, deps
├── requirements.txt           # Pip fallback deps
├── src/
│   ├── __init__.py
│   ├── index_and_retrieve.py  # Main script: index | search
│   ├── feature_extractor.py   # ViT-L/16 feature extraction
│   ├── retrieval_system.py    # FAISS index + metadata (ImageRetrievalSystem)
│   ├── image_index.faiss      # Saved FAISS index (generated)
│   └── image_metadata.json    # Index id → image path (generated)
├── utils/
│   ├── convert_images_to_jpg.py
│   ├── supported_image_types.json
│   └── yolov8_cropper.py
├── readme_files/
├── support_database_images/   # Default directory to index
└── query_images/              # Default directory for query images
```

---

## Requirements

- Python 3.10+
- [Hatch](https://hatch.pypa.io/latest/) (recommended) or pip
- Dependencies: see `pyproject.toml` or `requirements.txt` (torch, torchvision, faiss-cpu, Pillow, matplotlib; use `faiss-gpu` for GPU FAISS)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/rapido_image_ret.git
   cd rapido_image_ret
   ```

2. Using Hatch (recommended):

   ```bash
   pip install hatch
   hatch env create
   hatch shell
   ```

   or 

   ```
   source venv/bin/activate
   ```

   Dependencies are installed by the Hatch env. To install manually in the env:

   ```bash
   pip install -r requirements.txt
   ```

3. Or with pip only (from project root):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Run from project root so that `utils` and `src` are on the path (see `index_and_retrieve.py`).

## Usage

The main script `src/index_and_retrieve.py` supports two tasks: **index** and **search**. Run it from the **project root** so that `utils` and `retrieval_system` (in `src`) are importable.

### Step 1: Index images

Build the searchable database from an image directory. Images are converted to JPG (if needed) and then indexed.

1. In `src/index_and_retrieve.py`, set `task = "index"`.
2. Set `image_dir` to your folder (default: `support_database_images`).
3. Run from project root:

   ```bash
   python src/index_and_retrieve.py
   ```

   Or with Hatch:

   ```bash
   hatch run index
   ```

   This writes `src/image_index.faiss` and `src/image_metadata.json`.

### Step 2: Search

Retrieve the most similar images for a query image.

1. In `src/index_and_retrieve.py`, set `task = "search"`.
2. Set `query_image` to your query file (default uses `query_images/ray-ban-rb2132.jpg`).
3. Run from project root:

   ```bash
   python src/index_and_retrieve.py
   ```

   Results print to the console; if a query path is given, a plot shows the query and the closest match.

### Programmatic use

You can call the orchestration function directly:

```python
from src.index_and_retrieve import run_image_retrieval

# Index
run_image_retrieval(
    task="index",
    image_dir="/path/to/images",
    index_path="src/image_index.faiss",
    metadata_path="src/image_metadata.json",
    use_gpu=False,
)

# Search
run_image_retrieval(
    task="search",
    query_image="/path/to/query.jpg",
    index_path="src/image_index.faiss",
    metadata_path="src/image_metadata.json",
    num_results=5,
    use_gpu=False,
)
```
