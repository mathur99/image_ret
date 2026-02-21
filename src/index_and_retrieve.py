import sys
import os

# Set environment variables BEFORE importing torch to prevent OpenMP crashes on macOS M Series 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import logging
import math
from PIL import Image
import matplotlib.pyplot as plt

# Ensure utils and retrieval_system are found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.convert_images_to_jpg import convert_images_in_dir_to_jpg
from retrieval_system import ImageRetrievalSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_results(results, query_image_path=None):
    if not results:
        print("\nNo matches found!")
        return

    print("\nSearch Results:")
    print("-" * 50)
    for i, item in enumerate(results, 1):
        # unpack first two elements, ignore extras
        path, distance, *rest = item  
        similarity = 1.0 / (1.0 + distance)
        print(f"{i}. Image: {os.path.basename(path)}")
        print(f"   Full path: {path}")
        print(f"   Similarity Score: {similarity:.3f}")
        print(f"   Distance: {distance:.3f}")
        print("-" * 50)

    if query_image_path:
        query_img = Image.open(query_image_path)
        match_img = Image.open(results[0][0])
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Query Image")
        plt.imshow(query_img)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title("Closest Match")
        plt.imshow(match_img)
        plt.axis('off')
        plt.show()


def run_image_retrieval(task, image_dir=None, query_image=None, index_path="image_index.faiss",
                        metadata_path="image_metadata.json", num_results=5, use_gpu=False):
    try:
        if task == "index":
            if not image_dir:
                raise ValueError("image_dir is required for indexing task")

            converted_dir = os.path.join(image_dir, "converted_jpgs")
            supported_exts_json = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils", "supported_image_types.json"))
            logger.info(f"Converting supported images in {image_dir} to jpg format...")
            converted_images = convert_images_in_dir_to_jpg(image_dir, converted_dir, supported_exts_json)
            if not converted_images:
                raise ValueError("No images were converted.")

            logger.info(f"Number of converted images: {len(converted_images)}")
            retrieval_system = ImageRetrievalSystem(use_gpu=use_gpu, heavy_model=True)
            retrieval_system.index_images(converted_dir)
            retrieval_system.save(index_path, metadata_path)

        elif task == "search":
            if not query_image:
                raise ValueError("query_image is required for search task")
            retrieval_system = ImageRetrievalSystem(index_path=index_path, metadata_path=metadata_path, use_gpu=use_gpu, heavy_model=True)
            results = retrieval_system.search(query_image, k=num_results)
            print_results(results, query_image)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    task = "search" #search/index
    # Get project root directory (parent of src directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_dir = os.path.join(project_root, "support_database_images")
    query_folder = os.path.join(project_root, "query_images")
    query_image_filename = "ray-ban-rb2132.jpg"  # Just the filename, not full path
    index_path = os.path.join(os.path.dirname(__file__), "image_index.faiss")
    metadata_path = os.path.join(os.path.dirname(__file__), "image_metadata.json")

    if query_image_filename:
        query_image = os.path.join(query_folder, query_image_filename)
    else:
        files = [os.path.join(query_folder, f) for f in os.listdir(query_folder) if os.path.isfile(os.path.join(query_folder, f))]
        query_image = max(files, key=os.path.getctime) if files else None

    if task == "index":
        run_image_retrieval(task, image_dir=image_dir, index_path=index_path, metadata_path=metadata_path, use_gpu=False)
    elif task == "search" and query_image:
        run_image_retrieval(task, query_image=query_image, index_path=index_path, metadata_path=metadata_path, num_results=5, use_gpu=False)
