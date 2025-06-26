#!/usr/bin/env python3
"""
train_wave_model.py
-------------------
Main executable script for training the physics-informed wave forecasting model.
This script orchestrates data loading, model creation, training, and evaluation.
"""

import argparse
import torch
import json
from pathlib import Path

# Custom module imports with error handling
try:
    from enhanced_wave_sequencer import create_enhanced_dataloaders, FEATURE_NAMES
    from physics_wave_transformer import create_physics_model
    from physics_wave_trainer import PhysicsWaveTrainer
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Please ensure all required modules are in the same directory or your Python path.")
    exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Train Physics-Informed Wave Forecasting Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--features", type=str, default="enhanced_features.json",
                        help="Path to the enhanced features JSON file.")
    parser.add_argument("--sequence-length", type=int, default=24,
                        help="Input sequence length in hours.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: 'cuda', 'cpu', or 'auto'.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging.")
    parser.add_argument("--min-stations", type=int, default=3,
                        help="Minimum number of stations required per training sequence.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of worker processes for data loading.")
    args = parser.parse_args()

    # --- 1. Setup Environment ---
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("🌊 Physics-Informed Wave Forecasting Training")
    print("=" * 50)
    print(f"📂 Features file: {args.features}")
    print(f"⚙️  Device: {device}")
    print(f"📊 Sequence length: {args.sequence_length} hours")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"🏭 Min stations per sequence: {args.min_stations}")

    # --- 2. Load and Prepare Data ---
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"\n❌ Features file not found: {features_path}")
        return

    print("\n🔄 Creating dataloaders with 70/15/15 split...")
    try:
        train_loader, val_loader, test_loader, normalizer = create_enhanced_dataloaders(
            features_file=str(features_path),
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            split_ratios=(0.7, 0.15, 0.15),
            min_stations=args.min_stations,
            interpolate_missing=True,
            num_workers=args.num_workers
        )
        print(f"📊 Train batches: {len(train_loader)}")
        print(f"📊 Validation batches: {len(val_loader)}")
        print(f"🧪 Test batches: {len(test_loader)}")
        if len(train_loader) == 0:
            print("\n❌ No training sequences were created! Check data or parameters.")
            return

    except Exception as e:
        print(f"\n❌ Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 3. Create Model ---
    print("\n🧠 Creating physics-informed model...")
    try:
        model, criterion, optimizer, scheduler = create_physics_model(
            device=str(device)
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"🔢 Model parameters: {num_params:,}")
    except Exception as e:
        print(f"\n❌ Error creating model: {e}")
        return

    # --- 4. Train Model ---
    print("\n🏋️ Starting training...")
    try:
        trainer = PhysicsWaveTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=str(device),
            log_wandb=not args.no_wandb,
            feature_names=FEATURE_NAMES 
        )

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=args.epochs,
            normalizer=normalizer
        )
        print("\n🎉 Training completed successfully!")

    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 5. Save Artifacts and Report Results ---
    print("\n📊 Generating training curves...")
    trainer.plot_training_curves(save_path='training_curves.png')

    with open('training_history.json', 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        serializable_history = convert_numpy_types(history)
        json.dump(serializable_history, f, indent=4)

    print("\n✅ Results saved:")
    print(f"  📈 training_curves.png")
    print(f"  💾 {trainer.save_dir}/best_model.pth - Best model checkpoint")
    print(f"  💾 {trainer.save_dir}/wave_normalizer.pkl - Fitted data normalizer")
    print(f"  📊 training_history.json - Complete training metrics")

    if 'test_metrics' in history:
        print("\n" + "="*50)
        print("🧪 FINAL UNBIASED TEST SET PERFORMANCE")
        print("="*50)
        test_metrics = history['test_metrics']
        print(f"  - Test Loss: {test_metrics.get('test_total_loss', 0):.4f}")
        print(f"  - Test Wave Height MAE: {test_metrics.get('test_wave_height_mae', 0):.3f}m")
        print(f"  - Test Peak Period MAE: {test_metrics.get('test_peak_period_mae', 0):.3f}s")
        print(f"  - Test Energy Conservation Error: {test_metrics.get('test_energy_conservation_error', 0):.4f}")

if __name__ == "__main__":
    main()