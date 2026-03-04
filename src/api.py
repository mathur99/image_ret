import os
import tempfile

import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from .retrieval_system import ImageRetrievalSystem
from .segmenter import ImageSegmenter

app = FastAPI(title="Image Retrieval API")

INDEX_NAME = "image_index"

# loaded once at startup
system: ImageRetrievalSystem = None
segmenter: ImageSegmenter = None
cfg: dict = {}


def _index_file(directory, version):
    return os.path.join(directory, f"{INDEX_NAME}_v{version}.npz")


@app.on_event("startup")
def startup():
    global system, segmenter, cfg

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    index_dir = cfg["index_dir"]

    if "version" in cfg:
        load_path = _index_file(index_dir, cfg["version"])
    else:
        import glob as g

        pattern = os.path.join(index_dir, f"{INDEX_NAME}_v*.npz")
        files = g.glob(pattern)
        if not files:
            raise RuntimeError(f"No index found in {index_dir}")

        best_version, best_path = 0, None
        for f in files:
            try:
                v = int(os.path.splitext(os.path.basename(f))[0].split("_v")[-1])
                if v > best_version:
                    best_version, best_path = v, f
            except ValueError:
                continue
        load_path = best_path

    if not load_path or not os.path.exists(load_path):
        raise RuntimeError(f"Index not found: {load_path}")

    system = ImageRetrievalSystem(
        model=cfg.get("model", "vit_l_16"),
        index_path=load_path,
    )
    segmenter = ImageSegmenter()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
async def search(image: UploadFile = File(...)):
    if system is None:
        raise HTTPException(status_code=503, detail="System not loaded yet")

    suffix = os.path.splitext(image.filename or ".jpg")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await image.read()
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Empty file. Send multipart/form-data with field 'image' containing an image file (e.g. JPEG, PNG).",
            )
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()

        try:
            with Image.open(tmp.name) as img:
                img.verify()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or unsupported image. Use multipart/form-data, key 'image', and a valid image file (JPEG/PNG). Error: {e!s}",
            )

        top_k = cfg.get("top_k", 5)
        threshold = cfg.get("threshold", 0.5)

        results = system.search(tmp.name, k=top_k, segmenter=segmenter)

        matches = []
        for label, distance, date_added in results:
            if distance <= threshold:
                matches.append({
                    "label": label,
                    "distance": round(distance, 4),
                    "date_added": date_added[:10],
                })

        return {"matches": matches}
    finally:
        os.unlink(tmp.name)
