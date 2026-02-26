import os
import time

# must be set before torch import (macOS OpenMP fix)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import torch
import numpy as np
from torchvision import models
from PIL import Image

# model name → (constructor, weights, embedding dim)
MODELS = {
    "vit_b_32": (models.vit_b_32, models.ViT_B_32_Weights.IMAGENET1K_V1, 768),
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1, 768),
    "vit_l_32": (models.vit_l_32, models.ViT_L_32_Weights.IMAGENET1K_V1, 1024),
    "vit_l_16": (models.vit_l_16, models.ViT_L_16_Weights.IMAGENET1K_V1, 1024),
    "vit_h_14": (models.vit_h_14, models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1, 1280),
}


class ImageFeatureExtractor:
    """ViT embeddings, L2-normalized."""

    def __init__(self, model_name="vit_l_16"):
        if model_name not in MODELS:
            available = list(MODELS.keys())
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {available}")

        constructor, weights, self.feature_dim = MODELS[model_name]

        t0 = time.perf_counter()
        print(f"Loading {model_name} ({self.feature_dim}-d)...", end=" ", flush=True)
        self.model = constructor(weights=weights)
        # drop classifier head to get raw embeddings
        self.model.heads = torch.nn.Identity()
        self.model.eval()
        print(f"done ({time.perf_counter() - t0:.1f}s)")

        # preprocessing shipped with the weights
        self.preproc = weights.transforms()

    def extract(self, image):
        """L2-normalized embedding. Accepts a file path or PIL Image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        tensor = self.preproc(image).unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(tensor).flatten(1)

        embedding = embedding.cpu().numpy()

        # L2-normalize so inner product == cosine similarity
        norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-6
        embedding = embedding / norm

        return embedding.flatten()
