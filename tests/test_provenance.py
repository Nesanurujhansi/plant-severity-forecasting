"""Unit tests for perceptual image hashing and deduplication clustering."""

from PIL import Image
import numpy as np
from ml.data.provenance import compute_dhash, hamming_distance, cluster_near_duplicates

def test_compute_dhash():
    # Create synthetic gradient image
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        img_array[i, :, :] = i * 2
    img1 = Image.fromarray(img_array)
    
    hash1 = compute_dhash(img1)
    hash2 = compute_dhash(img1)
    
    assert isinstance(hash1, str)
    assert len(hash1) == 16
    assert hash1 == hash2

def test_hamming_distance():
    h1 = "000000000000000f"
    h2 = "0000000000000007" # Differs in 1 bit
    dist = hamming_distance(h1, h2)
    assert dist == 1

def test_cluster_near_duplicates():
    hashes = [
        "000000000000000f",
        "000000000000000e", # Distance <= 5
        "ffffffffffffffff"  # Far distance
    ]
    clusters = cluster_near_duplicates(hashes, max_distance=5)
    assert len(clusters) == 3
    assert clusters[0] == clusters[1]
    assert clusters[0] != clusters[2]
