#!/usr/bin/env python3
"""
Read LAION 400M dataset (webdataset type) /network/datasets/laion400m/laion400m/{00000..41407}.tar
Create a new dataset with only 80M random samples
Save the new dataset (webdataset type) to ~/scratch/data/laion400m/laion400m_80M/

Parallelized: splits shards across workers, each writes its own output shards,
then renumbers all output shards sequentially at the end.
"""

import os
import glob
import random
import shutil
import tempfile
from multiprocessing import Pool, cpu_count

import webdataset as wds

# Configuration
SRC_DIR = "/network/datasets/laion400m/laion400m/"
DST_DIR = os.path.expanduser("/home/mila/s/sarvjeet-singh.ghotra/scratch/data/laion400m/laion400m_40M/")
NUM_SAMPLES = 40_000_000
TOTAL_SAMPLES_APPROX = 41408 * 8000  # ~331M samples across all shards
SAMPLES_PER_SHARD = 8000  # Match input format (~8000 samples per shard)
SEED = 829
NUM_WORKERS = 16


def process_chunk(args):
    """Worker function: process a chunk of shards, write kept samples to worker-specific output dir."""
    worker_id, shard_paths, keep_prob, worker_seed, worker_out_dir = args

    random.seed(worker_seed)
    os.makedirs(worker_out_dir, exist_ok=True)

    output_pattern = os.path.join(worker_out_dir, "%05d.tar")
    num_kept = 0
    num_seen = 0

    dataset = wds.WebDataset(shard_paths).decode()

    with wds.ShardWriter(output_pattern, maxcount=SAMPLES_PER_SHARD) as sink:
        for sample in dataset:
            num_seen += 1

            if random.random() >= keep_prob:
                continue

            out = {"__key__": f"{num_kept:09d}"}
            for key, value in sample.items():
                if key in ("__key__", "__url__"):
                    continue
                out[key] = value
            sink.write(out)
            num_kept += 1

            if num_kept % 100_000 == 0:
                print(f"  [Worker {worker_id}] Kept {num_kept:,} / seen {num_seen:,}")

    print(f"  [Worker {worker_id}] Done: kept {num_kept:,} / seen {num_seen:,}")
    return worker_id, num_kept, num_seen


def main():
    rng = random.Random(SEED)

    # Create output directory
    os.makedirs(DST_DIR, exist_ok=True)

    # Get all source shard paths
    all_shards = sorted(glob.glob(os.path.join(SRC_DIR, "*.tar")))
    print(f"Found {len(all_shards)} source shards")

    # Shuffle shards for randomness, then split evenly across workers
    shuffled_shards = all_shards[:]
    rng.shuffle(shuffled_shards)

    chunks = [[] for _ in range(NUM_WORKERS)]
    for i, shard in enumerate(shuffled_shards):
        chunks[i % NUM_WORKERS].append(shard)

    keep_prob = NUM_SAMPLES / TOTAL_SAMPLES_APPROX
    print(f"Sampling probability: {keep_prob:.4f} ({keep_prob*100:.2f}%)")
    print(f"Using {NUM_WORKERS} workers, each processing ~{len(chunks[0])} shards")

    # Create a temp directory inside DST_DIR for worker outputs
    tmp_base = tempfile.mkdtemp(prefix="workers_", dir=DST_DIR)

    # Prepare worker args: each gets a unique seed derived from the main seed
    worker_args = []
    for wid in range(NUM_WORKERS):
        worker_out_dir = os.path.join(tmp_base, f"worker_{wid:02d}")
        worker_seed = rng.randint(0, 2**63)
        worker_args.append((wid, chunks[wid], keep_prob, worker_seed, worker_out_dir))

    # Run workers in parallel
    print("Starting parallel processing...")
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(process_chunk, worker_args)

    total_kept = sum(r[1] for r in results)
    total_seen = sum(r[2] for r in results)
    print(f"\nAll workers done. Total kept: {total_kept:,} / seen: {total_seen:,}")

    # Collect all worker shards and rename them sequentially into the final directory.
    # No re-encoding needed — just move the files.
    print("Renaming and moving worker shards to final directory...")
    all_worker_shards = []
    for wid in range(NUM_WORKERS):
        worker_out_dir = os.path.join(tmp_base, f"worker_{wid:02d}")
        worker_shards = sorted(glob.glob(os.path.join(worker_out_dir, "*.tar")))
        all_worker_shards.extend(worker_shards)

    for i, src_path in enumerate(all_worker_shards):
        dst_path = os.path.join(DST_DIR, f"{i:05d}.tar")
        shutil.move(src_path, dst_path)

    # Clean up temp worker directories
    shutil.rmtree(tmp_base)

    print(f"\nDone! {total_kept:,} samples saved to {DST_DIR}")

    # List output files with statistics
    output_files = sorted([f for f in os.listdir(DST_DIR) if f.endswith('.tar')])
    print(f"Created {len(output_files)} shard(s):")
    total_size = 0
    for f in output_files:
        fpath = os.path.join(DST_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        total_size += size_mb
    print(f"Total size: {total_size:.1f} MB ({total_size/1024:.1f} GB)")


if __name__ == "__main__":
    main()