#!/usr/bin/env python3
"""
train_all.py - Train all three models for the enhanced hCaptcha solver

This script:
1. Generates a large synthetic dataset (100+ classes with drawing effects)
2. Generates challenge rounds (point, drag, grid)
3. Trains TileNet, PointNet, and DragNet with enhanced parameters

Usage:
    python train_all.py                    # Full training
    python train_all.py --quick            # Quick test with fewer samples
    python train_all.py --tile-only        # Train only tile classifier
"""

import argparse
import os
import subprocess
import sys
import time

def run_cmd(cmd, desc):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"FAILED after {elapsed:.1f}s")
        return False
    print(f"SUCCESS in {elapsed:.1f}s")
    return True

def main():
    ap = argparse.ArgumentParser(description="Train all hCaptcha solver models")
    ap.add_argument("--quick", action="store_true", help="Quick test with fewer samples")
    ap.add_argument("--tile-only", action="store_true", help="Train only tile classifier")
    ap.add_argument("--generate-only", action="store_true", help="Only generate data, don't train")
    ap.add_argument("--classes", type=int, default=139, help="Number of classes to use")
    args = ap.parse_args()

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    data_v2_dir = os.path.join(script_dir, "data_v2")
    models_dir = os.path.join(script_dir, "models")
    
    # Determine sample counts
    if args.quick:
        per_class = 100
        n_point = 500
        n_drag = 300
        n_grid = 200
        epochs_tile = 3
        epochs_point = 3
        epochs_drag = 3
    else:
        per_class = 500  # 500 * 139 = ~70k images for tile training
        n_point = 3000
        n_drag = 2000
        n_grid = 1000
        epochs_tile = 12
        epochs_point = 10
        epochs_drag = 10

    os.makedirs(models_dir, exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     ENHANCED hCaptcha SOLVER TRAINING                        ║
║     139 Classes | Drawing Effects | High Accuracy            ║
╠══════════════════════════════════════════════════════════════╣
║  Per class samples:      {per_class:>6}                           ║
║  Point rounds:           {n_point:>6}                           ║
║  Drag rounds:            {n_drag:>6}                           ║
║  Grid rounds:            {n_grid:>6}                           ║
║  Tile epochs:            {epochs_tile:>6}                           ║
║  Point epochs:           {epochs_point:>6}                           ║
║  Drag epochs:            {epochs_drag:>6}                           ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Generate tile dataset
    print("\n[Step 1/6] Generating tile dataset with drawing effects...")
    cmd = [
        sys.executable, "make_dataset.py",
        "--per_class", str(per_class),
        "--out", data_dir,
        "--size", "96",
        "--seed", "1"
    ]
    if args.quick:
        cmd.append("--no_preview")
    if not run_cmd(cmd, "Generate Tile Dataset"):
        sys.exit(1)

    # Step 2: Generate challenge rounds
    print("\n[Step 2/6] Generating challenge rounds (point, drag, grid)...")
    os.makedirs(data_v2_dir, exist_ok=True)
    cmd = [
        sys.executable, "make_challenges.py",
        "--out", os.path.join(data_v2_dir, "challenges"),
        "--n_point", str(n_point),
        "--n_drag", str(n_drag),
        "--n_grid", str(n_grid),
        "--size", "96",
        "--seed", "7"
    ]
    if args.quick:
        cmd.append("--seed")  # Just use seed 7
    if not run_cmd(cmd, "Generate Challenge Rounds"):
        sys.exit(1)

    if args.generate_only:
        print("\n[OK] Data generation complete. Skipping training.")
        return

    # Step 3: Train TileNet
    print("\n[Step 3/6] Training TileNet (tile classifier)...")
    cmd = [
        sys.executable, "train_models.py",
        "--task", "tile",
        "--epochs", str(epochs_tile),
        "--batch", "64",
        "--size", "80",
        "--width", "32",
        "--lr", "5e-4",
        "--seed", "0",
        "--data", data_dir,
        "--models", models_dir,
        "--real_repeat", "50"
    ]
    if not run_cmd(cmd, "Train TileNet"):
        sys.exit(1)

    if args.tile_only:
        print("\n[OK] TileNet training complete.")
        return

    # Step 4: Train PointNet
    print("\n[Step 4/6] Training PointNet (point localization)...")
    challenges_path = os.path.join(data_v2_dir, "challenges", "manifest.jsonl")
    cmd = [
        sys.executable, "train_models.py",
        "--task", "point",
        "--epochs", str(epochs_point),
        "--batch", "48",
        "--size", "96",
        "--width", "24",
        "--lr", "5e-4",
        "--seed", "0",
        "--data", challenges_path,
        "--models", models_dir
    ]
    if not run_cmd(cmd, "Train PointNet"):
        sys.exit(1)

    # Step 5: Train DragNet
    print("\n[Step 5/6] Training DragNet (drag localization)...")
    cmd = [
        sys.executable, "train_models.py",
        "--task", "drag",
        "--epochs", str(epochs_drag),
        "--batch", "48",
        "--size", "96",
        "--width", "24",
        "--lr", "5e-4",
        "--seed", "0",
        "--data", challenges_path,
        "--models", models_dir
    ]
    if not run_cmd(cmd, "Train DragNet"):
        sys.exit(1)

    # Step 6: Summary
    print("\n[Step 6/6] Training Summary")
    print("=" * 60)
    print("Models saved to:", models_dir)
    for model in ["tile", "point", "drag"]:
        pt_path = os.path.join(models_dir, f"{model}.pt")
        json_path = os.path.join(models_dir, f"{model}.json")
        if os.path.exists(pt_path):
            size_mb = os.path.getsize(pt_path) / 1e6
            print(f"  {model}.pt: {size_mb:.1f} MB")
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     TRAINING COMPLETE - ENHANCED SOLVER READY               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
