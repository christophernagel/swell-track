#!/usr/bin/env python3
"""
train_wave_model.py
------------------
Quick start script for training your physics-informed wave forecasting model
using your enhanced 20-dimensional features.
"""

import argparse
import torch
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train Physics-Informed Wave Forecasting Model")
    parser.add_argument("--features", default="pacific_enhanced_features.json", 
                       help="Enhanced features JSON file from full_spectrum_processor")
    parser.add_argument("--sequence-length", type=int, default=24, 
                       help="Sequence length in hours (default: 24)")
    parser.add_argument("--batch-size", type=int, default=8, 
                       help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("--device", default="auto", 
                       help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--no-wandb", action="store_true", 
                       help="Disable Weights & Biases logging")
    parser.add_argument("--min-stations", type=int, default=3,
                       help="Minimum stations required per sequence")
    
    args = parser.parse_args()
    
    # Setup device
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
    
    # Check if features file exists
    if not Path(args.features).exists():
        print(f"❌ Features file not found: {args.features}")
        print("💡 Run this command first:")
        print(f"   python data/full_spectrum_processor.py --output {args.features}")
        return 1
    
    # Load and check features file
    try:
        with open(args.features, 'r') as f:
            features_data = json.load(f)
        
        num_stations = len(features_data)
        total_features = sum(len(data.get('enhanced_features', [])) for data in features_data.values())
        
        print(f"✅ Loaded features: {num_stations} stations, {total_features:,} feature vectors")
        
        # Show station coverage
        station_ids = list(features_data.keys())
        print(f"📍 Stations: {', '.join(station_ids)}")
        
    except Exception as e:
        print(f"❌ Error loading features file: {e}")
        return 1
    
    # Import training components
    try:
        from enhanced_wave_sequencer import create_enhanced_dataloaders
        from physics_wave_transformer import create_physics_model
        from physics_wave_trainer import PhysicsWaveTrainer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all training modules are in your Python path")
        return 1
    
    print("\n🔄 Creating dataloaders...")
    
    # Create enhanced dataloaders
    try:
        train_loader, val_loader = create_enhanced_dataloaders(
            features_file=args.features,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            split_ratio=0.8,
            min_stations=args.min_stations,
            normalize_features=True,
            interpolate_missing=True
        )
        
        print(f"📊 Train batches: {len(train_loader)}")
        print(f"📊 Validation batches: {len(val_loader)}")
        
        if len(train_loader) == 0:
            print("❌ No training sequences created!")
            print("💡 Try reducing --min-stations or --sequence-length")
            return 1
            
    except Exception as e:
        print(f"❌ Error creating dataloaders: {e}")
        return 1
    
    print("\n🧠 Creating physics-informed model...")
    
    # Create model
    try:
        model, criterion, optimizer, scheduler = create_physics_model(device=str(device))
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"🔢 Model parameters: {num_params:,}")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return 1
    
    print("\n🏋️ Starting training...")
    
    # Create trainer
    try:
        trainer = PhysicsWaveTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=str(device),
            log_wandb=not args.no_wandb
        )
        
        # Train the model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs
        )
        
        print("\n🎉 Training completed successfully!")
        
        # Plot training curves
        print("📊 Generating training curves...")
        trainer.plot_training_curves('training_curves.png')
        
        # Save training history
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n✅ Results saved:")
        print("  📈 training_curves.png - Training progress plots")
        print("  💾 wave_model_checkpoints/best_model.pth - Best model checkpoint")
        print("  📊 training_history.json - Complete training metrics")
        
        # Show final metrics
        if 'val_wave_height_mae' in history:
            final_mae = history['val_wave_height_mae'][-1]
            print(f"\n🌊 Final validation wave height MAE: {final_mae:.3f}m")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())