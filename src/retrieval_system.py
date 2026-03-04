import os
import gc
import time
from datetime import datetime, timezone
from collections import Counter

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

from .feature_extractor import ImageFeatureExtractor


class ImageRetrievalSystem:
    """FAISS cosine-similarity index over ViT embeddings, stored as .npz."""

    def __init__(self, model="vit_l_16", index_path=None):
        self.extractor = ImageFeatureExtractor(model_name=model)
        dim = self.extractor.feature_dim

        self.features = np.empty((0, dim), dtype=np.float32)
        self.labels = np.array([], dtype=str)
        self.dates_added = np.array([], dtype=str)
        self.dates_edited = np.array([], dtype=str)

        # inner product on L2-normed vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dim)

        if index_path:
            self.load(index_path)

    def index_images(self, directory, segmenter, crops_dir, supported_extensions):
        # find all supported image files
        filenames = []
        for f in sorted(os.listdir(directory)):
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_extensions:
                filenames.append(f)

        # label = filename without extension
        labels = []
        for f in filenames:
            name = os.path.splitext(f)[0]
            labels.append(name)

        print(f"Found {len(filenames)} images")

        # check for duplicate labels
        counts = Counter(labels)
        duplicates = []
        for name, count in counts.items():
            if count > 1:
                duplicates.append(name)
        if duplicates:
            raise ValueError(f"Duplicate filenames: {duplicates}")

        # detect objects in each image
        items = self._detect_objects(directory, filenames, labels, segmenter, crops_dir)
        if not items:
            raise ValueError("Nothing to index.")

        # extract embeddings for each crop
        timestamp = datetime.now(timezone.utc).isoformat()
        new_features = []
        new_labels = []

        t0 = time.perf_counter()
        for label, image in tqdm(items, desc="Embedding"):
            try:
                feature = self.extractor.extract(image)
                new_features.append(feature)
                new_labels.append(label)
            except Exception as error:
                print(f"  Failed on {label}: {error}")

        if not new_features:
            raise ValueError("No features extracted.")
        print(f"Embedded {len(new_features)} entries ({time.perf_counter() - t0:.1f}s)")

        # normalize and add to index
        feature_matrix = np.stack(new_features).astype(np.float32)
        faiss.normalize_L2(feature_matrix)

        if self.features.size:
            self.features = np.vstack([self.features, feature_matrix])
        else:
            self.features = feature_matrix

        self.labels = np.concatenate([self.labels, new_labels])
        self.dates_added = np.concatenate([self.dates_added, [timestamp] * len(new_labels)])
        self.dates_edited = np.concatenate([self.dates_edited, [timestamp] * len(new_labels)])
        self.index.add(feature_matrix)

    def _detect_objects(self, directory, filenames, labels, segmenter, crops_dir):
        """Run object detection on each image, return (label, PIL image) pairs."""
        os.makedirs(crops_dir, exist_ok=True)
        items = []

        t0 = time.perf_counter()
        for filename, label in tqdm(zip(filenames, labels), total=len(filenames), desc="Detecting"):
            path = os.path.join(directory, filename)
            try:
                crops = segmenter.segment(path)

                for i, crop in enumerate(crops):
                    if len(crops) > 1:
                        crop_label = f"{label}_obj{i}"
                    else:
                        crop_label = label

                    crop.save(os.path.join(crops_dir, f"{crop_label}.jpg"), "JPEG")
                    items.append((crop_label, crop))

            except Exception as error:
                print(f"  Detection failed for {filename}: {error}")

        print(f"Detected {len(items)} objects ({time.perf_counter() - t0:.1f}s)")

        # free detection model before ViT runs
        del segmenter
        gc.collect()

        return items

    def search(self, query_path, k=5, segmenter=None, query_crops_dir=None):
        """Returns [(label, cosine_distance, date_added), ...] sorted by distance ascending."""
        if segmenter:
            images = segmenter.segment(query_path)
            del segmenter
            gc.collect()
        else:
            images = [Image.open(query_path).convert("RGB")]

        if query_crops_dir:
            os.makedirs(query_crops_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(query_path))[0]
            for i, img in enumerate(images):
                tag = f"{stem}_obj{i}" if len(images) > 1 else stem
                img.save(os.path.join(query_crops_dir, f"{tag}.jpg"), "JPEG")
            print(f"Saved {len(images)} query crop(s) to {query_crops_dir}")

        # search each crop, keep best similarity per label
        best_scores = {}

        for image in images:
            feature = self.extractor.extract(image).reshape(1, -1)
            faiss.normalize_L2(feature)

            similarities, indices = self.index.search(feature, k)

            for idx, sim in zip(indices[0], similarities[0]):
                if idx >= len(self.labels):
                    continue

                label = self.labels[idx]
                score = float(sim)

                if label not in best_scores or score > best_scores[label][0]:
                    date_added = str(self.dates_added[idx])
                    best_scores[label] = (score, date_added)

        # convert to (label, cosine_distance, date_added), sorted by distance ascending
        results = []
        for label, (similarity, date_added) in best_scores.items():
            distance = 1.0 - similarity
            results.append((label, distance, date_added))

        results.sort(key=lambda x: x[1])
        return results[:k]

    def save(self, path, version=1):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            features=self.features,
            label_name=self.labels,
            date_added=self.dates_added,
            date_edited=self.dates_edited,
            faiss_version=np.array(faiss.__version__),
            version=np.array(version),
        )
        print(f"Saved {len(self.labels)} entries (v{version}) -> {path}")

    def load(self, path):
        data = np.load(path, allow_pickle=False)

        self.features = data['features'].astype(np.float32)
        self.labels = data['label_name']
        self.dates_added = data['date_added']
        self.dates_edited = data['date_edited']

        # rebuild FAISS from stored features
        dim = self.features.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.features)

        version = int(data['version']) if 'version' in data else "?"
        print(f"Loaded {self.index.ntotal} entries (v{version}) from {path}")

        stored_faiss = str(data['faiss_version'])
        if stored_faiss != faiss.__version__:
            print(f"  Warning: index built with faiss {stored_faiss}, running {faiss.__version__}")
