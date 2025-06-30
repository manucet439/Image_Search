import os
import gc
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import chromadb
from chromadb.config import Settings
import json
import pickle
import psutil
import time

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # GB

# Cell 3: Configuration for Test Mode
TEST_MODE = False  #True
TEST_SIZE = 500  # Process only 500 images
print(f"🧪 TEST MODE ENABLED: Will process {TEST_SIZE} images only")
print(f"Initial memory usage: {get_memory_usage():.2f} GB")

# Cell 4: Setup CLIP Model
print("\nLoading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("⚠️  WARNING: Using CPU. Processing will be slower.")
    print("   Estimated time: ~30-45 minutes for 500 images")
else:
    print("✅ GPU detected! Processing will be fast.")
    print("   Estimated time: ~3-5 minutes for 500 images")

# Load model
model_name = "openai/clip-vit-base-patch32"
print(f"\nDownloading {model_name}...")

try:
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    print("Retrying with force_download...")
    model = CLIPModel.from_pretrained(model_name, force_download=True)
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained(model_name, force_download=True)
    model.eval()

print(f"Memory after model loading: {get_memory_usage():.2f} GB")

# Cell 5: Setup ChromaDB
print("\nSetting up ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db_test")

# Create fresh collection for test
collection_name = "image_embeddings_test_500"
try:
    client.delete_collection(name=collection_name)
except:
    pass

collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)
print(f"✅ Created collection: {collection_name}")

# Cell 6: Load Image Files
# Update this path to your image directory
current_working_directory = os.getcwd()
#IMAGE_DIR = "/content/images"  # UPDATE THIS PATH
IMAGE_DIR = os.path.join(current_working_directory, 'images')

print(f"\nScanning directory: {IMAGE_DIR}")
try:
    all_files = os.listdir(IMAGE_DIR)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    image_files = [f for f in all_files if f.lower().endswith(image_extensions)]

    total_available = len(image_files)
    print(f"Found {total_available} total images in directory")

    # Limit to TEST_SIZE
    if TEST_MODE:
        image_files = image_files[:TEST_SIZE]
        print(f"📌 Limited to {len(image_files)} images for testing")

except Exception as e:
    print(f"❌ Error accessing directory: {e}")
    print("Please update IMAGE_DIR path in Cell 6")
    raise
# Cell 7: Image Processing Functions
def load_and_preprocess_image(image_path):
    """Load and preprocess image for CLIP"""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading {os.path.basename(image_path)}: {e}")
        return None

@torch.no_grad()
def generate_image_embedding(image, model, processor, device):
    """Generate CLIP embedding for an image"""
    inputs = processor(images=image, return_tensors="pt").to(device)

    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    embedding = image_features.cpu().numpy().flatten()

    if device == "cuda":
        torch.cuda.empty_cache()

    return embedding
# Cell 8: Process Images
print(f"\n🚀 Starting processing of {len(image_files)} images...")
start_time = time.time()

embeddings = []
metadata_list = []
ids = []
failed_images = []

# Process with progress bar
for idx, filename in enumerate(tqdm(image_files, desc="Processing images")):
    image_path = os.path.join(IMAGE_DIR, filename)

    try:
        # Load and process image
        image = load_and_preprocess_image(image_path)

        if image is not None:
            # Generate embedding
            embedding = generate_image_embedding(image, model, processor, device)

            # Store results
            embeddings.append(embedding.tolist())
            metadata_list.append({
                "filename": filename,
                "path": image_path,
                "index": idx
            })
            ids.append(f"img_{idx:04d}")

            # Clean up
            del image

    except Exception as e:
        print(f"\n❌ Failed: {filename} - {e}")
        failed_images.append(filename)
        continue

    # Show progress every 100 images
    if (idx + 1) % 100 == 0:
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed
        print(f"\n  Processed: {idx + 1}/{len(image_files)} | Speed: {rate:.1f} img/sec | Memory: {get_memory_usage():.2f} GB")
# Cell 9: Store in ChromaDB
print(f"\n💾 Storing {len(embeddings)} embeddings in ChromaDB...")

# Add to ChromaDB in batches
batch_size = 100
for i in tqdm(range(0, len(embeddings), batch_size), desc="Storing in DB"):
    batch_end = min(i + batch_size, len(embeddings))

    collection.add(
        embeddings=embeddings[i:batch_end],
        metadatas=metadata_list[i:batch_end],
        ids=ids[i:batch_end]
    )

# Calculate statistics
total_time = time.time() - start_time
success_count = len(embeddings)
fail_count = len(failed_images)

print(f"\n{'='*60}")
print(f"✅ PROCESSING COMPLETE!")
print(f"{'='*60}")
print(f"📊 Successfully processed: {success_count}/{len(image_files)} images")
print(f"❌ Failed: {fail_count} images")
print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"🚀 Average speed: {success_count/total_time:.1f} images/second")
print(f"💾 Database location: ./chroma_db_test")

