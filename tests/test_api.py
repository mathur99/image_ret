"""Lightweight API tests. Run with: pytest tests/test_api.py -v"""
import io
from io import StringIO
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client():
    """Mock startup so tests run without real config/index."""
    import builtins

    mock_system = Mock()
    mock_system.search.return_value = []

    def fake_open(path, *args, **kwargs):
        if path == "config.yaml":
            return StringIO(
                "index_dir: data/index\nmodel: vit_b_32\ntop_k: 5\nthreshold: 0.5\nversion: 1\n"
            )
        return builtins.open(path, *args, **kwargs)

    with (
        patch("src.api.open", side_effect=fake_open),
        patch("src.api.os.path.exists", return_value=True),
        patch("src.api.ImageRetrievalSystem", return_value=mock_system),
        patch("src.api.ImageSegmenter", return_value=Mock()),
    ):
        from src.api import app

        with TestClient(app) as c:
            yield c


def _make_png_bytes():
    img = Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_empty_file(client):
    response = client.post("/search", files={"image": ("x.png", b"", "image/png")})
    assert response.status_code == 400
    assert "Empty" in response.json().get("detail", "")


def test_search_invalid_image(client):
    response = client.post(
        "/search",
        files={"image": ("x.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid" in response.json().get("detail", "")


def test_search_image_over_20mb(client):
    over_20mb = b"x" * (20 * 1024 * 1024 + 1)
    response = client.post(
        "/search",
        files={"image": ("large.png", over_20mb, "image/png")},
    )
    assert response.status_code == 413
    assert "20 MB" in response.json().get("detail", "")


def test_search_success(client):
    mock_system = Mock()
    mock_system.search.return_value = [
        ("label1", 0.2, "2024-01-01T00:00:00"),
    ]
    mock_segmenter = Mock()
    png_bytes = _make_png_bytes()

    with patch("src.api.system", mock_system), patch("src.api.segmenter", mock_segmenter):
        response = client.post(
            "/search",
            files={"image": ("x.png", png_bytes, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "matches" in data
    assert isinstance(data["matches"], list)
