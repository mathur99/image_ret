import os

# Set environment variables BEFORE importing torch to prevent OpenMP crashes on macOS
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import torch
import numpy as np
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torchvision import transforms
from PIL import Image

class ImageFeatureExtractor:
    """
    Extracts L2-normalized ViT embeddings from the full image.
    """
    def __init__(self, heavy_model=True, device=None):
        # Select device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained ViT-L/16 and strip classifier head
        self.model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).to(self.device)
        self.model.heads = torch.nn.Identity()
        self.model.eval()
        self.feature_dim = 1024

        # Preprocessing: resize -> center-crop -> normalize
        self.preproc = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_path):
        """
        Returns a (feature_dim,) numpy vector L2-normalized for cosine similarity.
        """
        img = Image.open(image_path).convert("RGB")

        # Use the full image without cropping
        tensor = self.preproc(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor).flatten(1)  # shape (1, feature_dim)

        # Convert to numpy and L2-normalize
        feat_np = feat.cpu().numpy()
        feat_np /= (np.linalg.norm(feat_np, axis=1, keepdims=True) + 1e-6)
        return feat_np.flatten()
