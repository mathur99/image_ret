import logging
import os
import tempfile
import time

import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .retrieval_system import ImageRetrievalSystem
from .segmenter import ImageSegmenter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("image_ret.api")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Image Retrieval API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s %s %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response

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

    try:
        logger.info("Loading config from config.yaml")
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

        index_dir = cfg["index_dir"]
        model = cfg.get("model", "vit_l_16")

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

        logger.info("Index path: %s, model: %s", load_path, model)
        system = ImageRetrievalSystem(
            model=model,
            index_path=load_path,
        )
        segmenter = ImageSegmenter()
        logger.info("Startup complete")
    except Exception as e:
        logger.exception("Startup failed: %s", e)
        raise


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
@limiter.limit("5/minute")
async def search(request: Request, image: UploadFile = File(...)):
    if system is None:
        logger.warning("search: system not loaded (503)")
        raise HTTPException(status_code=503, detail="System not loaded yet")

    suffix = os.path.splitext(image.filename or ".jpg")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await image.read()
        if not content:
            logger.warning("search: empty file rejected")
            raise HTTPException(
                status_code=400,
                detail="Empty file. Send multipart/form-data with field 'image' containing an image file (e.g. JPEG, PNG).",
            )
        MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB
        if len(content) > MAX_IMAGE_BYTES:
            logger.warning("search: image over 20 MB rejected (size=%s)", len(content))
            raise HTTPException(
                status_code=413,
                detail="Image must be at most 20 MB.",
            )
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()

        try:
            with Image.open(tmp.name) as img:
                img.verify()
        except Exception as e:
            logger.warning("search: invalid image rejected: %s", e)
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
