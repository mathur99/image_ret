import time
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.transforms.functional import to_tensor


class ImageSegmenter:
    """Detects and crops objects using Faster R-CNN (MobileNetV3)."""

    def __init__(self, conf_thresh=0.3):
        self.conf_thresh = conf_thresh

        t0 = time.perf_counter()
        print("Loading Faster R-CNN...", end=" ", flush=True)
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        self.model.eval()
        print(f"done ({time.perf_counter() - t0:.1f}s)")

    @torch.no_grad()
    def segment(self, image_path, min_area_ratio=0.01, max_area_ratio=0.95):
        """Return cropped bounding boxes as PIL Images. Falls back to full image."""
        img = Image.open(image_path).convert("RGB")
        total_area = img.width * img.height

        tensor = to_tensor(img).unsqueeze(0)
        preds = self.model(tensor)[0]

        crops = []
        for box, score in zip(preds['boxes'], preds['scores']):
            if score < self.conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            ratio = area / total_area
            # skip tiny noise boxes and boxes that cover the whole image
            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))

        if not crops:
            crops.append(img)

        return crops
