<div align="center">

# Image Retrieval System

**Find lost items by visual similarity — snap a photo, get matches instantly.**

![header](.github/assets/header.webp)

Built with **PyTorch** / **torchvision** · Powered by **ViT** + **Faster R-CNN** + **FAISS**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/badge/pkg-uv-blueviolet)](https://docs.astral.sh/uv/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)](https://fastapi.tiangolo.com/)

</div>

---

**What it does** — Drop images of everyday items (shoes, wallets, phones, cards, bags) into a database folder, build an index, then search with a new photo. The system detects individual objects in every image, embeds them with a Vision Transformer, and finds the closest matches using cosine distance.

**Why it exists** — Lost-and-found desks deal with hundreds of items. This turns "does anyone recognize this?" into a quantified visual search.

### Key features

- **Zero CLI arguments** — everything lives in one `config.yaml`
- **Automatic object detection** — Faster R-CNN segments every image into individual object crops before embedding
- **Versioned indexes** — `image_index_v1.npz`, `v2`, `v3`... never overwrite, always reproducible
- **Configurable ViT backbone** — swap between 5 model sizes (768-d to 1280-d) with one line
- **Visual results** — matplotlib popup showing query vs matched crops with cosine distance and date
- **REST API** — FastAPI endpoint for image upload and search over HTTP (JSON response)
- **API safeguards** — 20 MB max upload size, rate limit (5 requests/min per client), request logging
- **Docker-ready** — single `docker compose up` to run the API in a container
- **Tests** — lightweight pytest suite for health and search (no heavy model load in CI)

![example](.github/assets/Figure_1.png)

---

## Architecture

> Full interactive version: open `architecture.excalidraw.json` in [Excalidraw](https://excalidraw.com/)

```mermaid
flowchart TB
    subgraph CONFIG["config.yaml"]
        C1["mode · model · threshold · top_k · index_dir"]
        C2["database_dir · query_image · extensions · version"]
    end

    subgraph INDEX["INDEX Pipeline"]
        direction LR
        I1["DB images"]
        I2["Faster R-CNN"]
        I3["ViT embed"]
        I4["FAISS index"]
        I5[".npz save"]
        I1 --> I2 --> I3 --> I4 --> I5
    end

    subgraph SEARCH["SEARCH Pipeline"]
        direction LR
        S1["Query"]
        S2["Segment"]
        S3["Embed"]
        S4["FAISS search"]
        S5["CLI popup / API JSON"]
        S1 --> S2 --> S3 --> S4 --> S5
    end

    CONFIG -.-> INDEX
    CONFIG -.-> SEARCH
    I5 -.-> S4

    style CONFIG fill:#fff9db,stroke:#fab005,color:#000
    style INDEX fill:#ebfbee,stroke:#40c057,color:#000
    style SEARCH fill:#d0ebff,stroke:#228be6,color:#000
```

### .npz index contents

```mermaid
classDiagram
    class image_index_vN_npz {
        float32[N × dim] features
        str[N] label_name
        str[N] date_added
        str[N] date_edited
        str faiss_version
        int version
    }
```

---

## How it works

1. **Index** — source images are segmented into object crops (Faster R-CNN), embedded with a Vision Transformer, L2-normalized, and stored in a versioned FAISS index (`.npz`).
2. **Search (CLI)** — a query image goes through the same segment-then-embed pipeline; FAISS finds the closest embeddings by cosine distance and displays passing matches in a matplotlib popup.
3. **Search (API)** — same pipeline exposed as a `POST /search` endpoint via FastAPI. Upload an image, get JSON results back.

Both pipelines are driven entirely by `config.yaml` — no CLI arguments, no argparse.

---

## Setup

```bash
git clone https://github.com/yourusername/image_ret.git
cd image_ret
./setup.sh
```

The setup script installs `uv` (if missing), runs `uv sync`, and extracts sample images from `data.zip` into `data/database/` and `data/query/`.

<details>
<summary>Manual setup (without script)</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if you have uv
uv sync
unzip data.zip                                      # extract sample images
```

</details>

## Usage

Edit `config.yaml`, then run:

```bash
uv run image-ret
```

### Indexing

Set `mode: index`. Images from `database_dir` are segmented, embedded, and saved to `index_dir` as a versioned `.npz` file.

- If `version` is set (e.g. `version: 1`), the index is saved as `image_index_v1.npz`.
- If `version` is omitted, it auto-increments from the latest existing version.

### Searching

Set `mode: search`. The query image is segmented, embedded, and compared against the loaded index.

- If `version` is set, that specific index is loaded (e.g. `image_index_v2.npz`).
- If `version` is omitted, the latest index in `index_dir` is loaded.
- Only matches with cosine distance **<= threshold** are shown.
- Results pop up in a matplotlib window showing the query alongside matched crops, with cosine distance and date added.

### API

The FastAPI server exposes the search pipeline over HTTP. It loads the index and models once at startup (using `config.yaml`), then accepts image uploads.

**Limits and behaviour**

- **Upload size** — Images must be at most **20 MB**. Larger uploads receive `413 Payload Too Large`.
- **Rate limit** — **5 requests per minute** per client (by IP). Exceeding returns `429 Too Many Requests`.
- **Logging** — Startup logs config path, index path, and model; each request is logged with method, path, status code, and duration. Rejected requests (empty file, invalid image, over size) are logged at warning level.

```bash
# run locally
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
# search
curl -X POST http://localhost:8000/search -F "image=@data/query/photo.jpg"
```

Response:

```json
{
  "matches": [
    { "label": "wallet_1_obj0", "distance": 0.312, "date_added": "2026-02-26" }
  ]
}
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok"}` |
| `/search` | POST | Multipart image upload (field `image`), returns matching items as JSON. Max 20 MB, 5 req/min per client. |

| Status | Meaning |
|---|---|
| 200 | Success, `matches` in body |
| 400 | Empty file or invalid/unsupported image |
| 413 | Image larger than 20 MB |
| 429 | Rate limit exceeded (5/min) |
| 503 | Server still starting or index not loaded |

### Docker

Build and run the API in a container. Data and config are mounted as volumes so the index doesn't need to be baked into the image.

```bash
docker compose up --build
```

This builds the image (Python 3.12 + uv), installs all dependencies, and starts the FastAPI server on port 8000. First run downloads model weights (~1.2 GB ViT + 74 MB Faster R-CNN).

To run in the background:

```bash
docker compose up -d          # start
docker compose logs -f        # follow logs
docker compose down           # stop
```

<details>
<summary>Manual Docker build (without compose)</summary>

```bash
docker build -t image-ret .
docker run -p 8000:8000 -v ./data:/app/data -v ./config.yaml:/app/config.yaml image-ret
```

</details>

### Tests

Lightweight API tests run without a real index (startup is mocked). Install dev deps and run:

```bash
uv sync --extra dev
uv run pytest tests/test_api.py -v
```

Tests cover: health check, search with valid image (mocked), empty/invalid image → 400, image over 20 MB → 413. Logs are shown during test runs (pytest `log_cli` in `pyproject.toml`).

---

## Config reference

```yaml
mode: index               # "index" or "search"

database_dir: data/database
query_image: data/query/photo.jpg

index_dir: data/index     # directory for versioned .npz files
top_k: 3
threshold: 0.5            # cosine distance — lower = more similar; <= threshold pass

version: 1                # specific index version (omit to auto-detect)

supported_extensions:
  - .png
  - .jpg
  - .jpeg
  - .webp

model: vit_l_16
```

### Available models

| Name | Embedding dim | Use case |
|---|---|---|
| `vit_b_32` | 768 | Fastest — prototyping, large datasets |
| `vit_b_16` | 768 | Balanced speed and accuracy |
| `vit_l_32` | 1024 | Higher quality, still reasonable speed |
| `vit_l_16` | 1024 | Best accuracy for most use cases **(recommended)** |
| `vit_h_14` | 1280 | Highest quality, slowest, needs most RAM |

---

## Index format (`.npz`)

Each versioned index file (`image_index_vN.npz`) contains:

| Key | Shape / Type | Description |
|---|---|---|
| `features` | `(N, dim)` float32 | L2-normalized ViT embeddings |
| `label_name` | `(N,)` str | Crop label (e.g. `wallet_1_obj0`) |
| `date_added` | `(N,)` str | ISO timestamp when indexed |
| `date_edited` | `(N,)` str | ISO timestamp of last edit |
| `faiss_version` | scalar str | FAISS library version used to build the index |
| `version` | scalar int | Index version number (v1, v2, ...) |

Use the inspection notebook (`notebooks/inspect_index.ipynb`) to browse index contents as a pandas DataFrame.

---

## Project structure

```
image_ret/
├── config.yaml                      # all configuration lives here
├── pyproject.toml                   # dependencies & entry point
├── setup.sh                         # one-command setup (install + unzip)
├── data.zip                         # sample images archive (database + query)
├── architecture.excalidraw.json     # full interactive architecture diagram
├── Dockerfile                       # Python 3.12 + uv container image
├── docker-compose.yaml              # one-command container setup
├── .dockerignore                    # keeps build context small
├── src/
│   ├── index_and_retrieve.py        # CLI entry point — reads config, runs index or search
│   ├── api.py                       # FastAPI server — POST /search, GET /health, logging, rate limit, 20 MB cap
│   ├── feature_extractor.py         # ViT embeddings (configurable model)
│   ├── retrieval_system.py          # FAISS index, .npz save/load, search
│   └── segmenter.py                 # Faster R-CNN MobileNetV3-FPN object detection
├── tests/
│   └── test_api.py                  # Lightweight API tests (health, search validation, 413)
├── notebooks/
│   └── inspect_index.ipynb          # browse .npz index contents in pandas
└── data/                            # created by setup.sh / unzip
    ├── database/                    # source images (png, jpg, webp)
    ├── query/                       # query images
    └── index/                       # generated: versioned .npz + crops/
```

## Models used

| Model | Purpose | Details |
|---|---|---|
| **Faster R-CNN MobileNetV3-FPN** | Object detection | Pre-trained on COCO. Crops bounding boxes with confidence > 0.3, filters tiny and full-image boxes. Runs on both index and query images. |
| **Vision Transformer (ViT)** | Feature extraction | Pre-trained on ImageNet. Classifier head removed for raw embeddings. L2-normalized so inner product = cosine similarity. |
| **FAISS IndexFlatIP** | Similarity search | Exact inner-product search on normalized vectors. Cosine distance = 1 − similarity. |

## Tech stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.10+, CPU-only |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Deep learning | PyTorch, torchvision |
| Search | FAISS (faiss-cpu) |
| API | FastAPI, uvicorn |
| Container | Docker, Docker Compose |
| Visualization | matplotlib |
| Config | YAML (`pyyaml`) |
| Inspection | Jupyter notebook, pandas |

---

## License

MIT
