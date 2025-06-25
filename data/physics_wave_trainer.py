"""
physics_wave_trainer.py
-----------------------
Training pipeline for physics-informed wave forecasting with your enhanced features.
Includes physics constraints, evaluation metrics, and model management.
"""

import torch
import torch.nn as nn
import numpy as np
import wandb
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class PhysicsWaveTrainer:
    """
    Enhanced trainer for physics-informed wave forecasting.
    Handles multi-station training with proper physics constraints.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        log_wandb: bool = True,
        save_dir: str = 'wave_model_checkpoints'
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_wandb = log_wandb
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training metrics
        self.metrics_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 15
        
        # Feature names for detailed analysis
        self.feature_names = [
            'sig_height', 'peak_period', 'mean_period', 'peak_freq', 'total_energy',
            'spectral_width', 'swell_energy', 'windsea_energy', 'swell_fraction', 'windsea_fraction',
            'primary_direction', 'primary_spread', 'secondary_direction', 'bimodal_strength', 'directional_separation',
            'wind_speed', 'wind_direction', 'wind_wave_alignment',
            'spectral_hs_error', 'spectral_tp_error'
        ]
        
        if self.log_wandb:
            wandb.init(
                project="physics-wave-forecasting",
                config={
                    "model_type": "PhysicsInformedWaveTransformer",
                    "features": "20D_enhanced_physics",
                    "architecture": str(model),
                    "criterion": str(criterion)
                }
            )
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch with physics-informed loss"""
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            sequence_features = batch['sequence_features'].to(self.device)
            target_features = batch['target_features'].to(self.device)
            station_coords = batch['station_coords'].to(self.device)
            station_masks = batch['station_masks'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(
                sequence_features=sequence_features,
                station_coords=station_coords,
                station_mask=station_masks,
                output_physics_separate=True
            )
            
            # Compute physics-informed loss
            losses = self.criterion(predictions, target_features, station_masks)
            total_loss = losses['total']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            for loss_name, loss_value in losses.items():
                epoch_metrics[f'train_{loss_name}_loss'].append(loss_value.item())
            
            # Physics validation metrics
            physics_metrics = self._compute_physics_metrics(predictions, target_features, station_masks)
            for metric_name, metric_value in physics_metrics.items():
                epoch_metrics[f'train_{metric_name}'].append(metric_value)
            
            # Log batch metrics
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss = {total_loss.item():.4f}, "
                      f"Wave Height MAE = {physics_metrics.get('wave_height_mae', 0):.3f}m")
        
        # Average metrics
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_metrics = defaultdict(list)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                sequence_features = batch['sequence_features'].to(self.device)
                target_features = batch['target_features'].to(self.device)
                station_coords = batch['station_coords'].to(self.device)
                station_masks = batch['station_masks'].to(self.device)
                
                # Forward pass
                predictions = self.model(
                    sequence_features=sequence_features,
                    station_coords=station_coords,
                    station_mask=station_masks,
                    output_physics_separate=True
                )
                
                # Compute loss
                losses = self.criterion(predictions, target_features, station_masks)
                
                # Track metrics
                for loss_name, loss_value in losses.items():
                    epoch_metrics[f'val_{loss_name}_loss'].append(loss_value.item())
                
                # Physics validation metrics
                physics_metrics = self._compute_physics_metrics(predictions, target_features, station_masks)
                for metric_name, metric_value in physics_metrics.items():
                    epoch_metrics[f'val_{metric_name}'].append(metric_value)
                
                # Collect predictions for detailed analysis
                valid_preds = predictions['combined'][station_masks].cpu().numpy()
                valid_targets = target_features[station_masks].cpu().numpy()
                all_predictions.append(valid_preds)
                all_targets.append(valid_targets)
        
        # Compute detailed validation metrics
        if all_predictions:
            all_predictions = np.vstack(all_predictions)
            all_targets = np.vstack(all_targets)
            detailed_metrics = self._compute_detailed_metrics(all_predictions, all_targets)
            epoch_metrics.update({f'val_{k}': [v] for k, v in detailed_metrics.items()})
        
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def _compute_physics_metrics(self, predictions: Dict[str, torch.Tensor], 
                                targets: torch.Tensor, station_mask: torch.Tensor) -> Dict[str, float]:
        """Compute physics-based validation metrics"""
        metrics = {}
        
        # Get valid predictions and targets
        pred_combined = predictions['combined'][station_mask]
        valid_targets = targets[station_mask]
        
        if len(pred_combined) == 0:
            return metrics
        
        # Wave height accuracy (most important metric)
        height_pred = pred_combined[:, 0].cpu().numpy()
        height_target = valid_targets[:, 0].cpu().numpy()
        
        valid_height_mask = ~np.isnan(height_target)
        if np.any(valid_height_mask):
            height_mae = np.mean(np.abs(height_pred[valid_height_mask] - height_target[valid_height_mask]))
            height_rmse = np.sqrt(np.mean((height_pred[valid_height_mask] - height_target[valid_height_mask])**2))
            height_mape = np.mean(np.abs((height_pred[valid_height_mask] - height_target[valid_height_mask]) / 
                                       np.maximum(height_target[valid_height_mask], 0.1))) * 100
            
            metrics.update({
                'wave_height_mae': height_mae,
                'wave_height_rmse': height_rmse,
                'wave_height_mape': height_mape
            })
        
        # Peak period accuracy
        period_pred = pred_combined[:, 1].cpu().numpy()
        period_target = valid_targets[:, 1].cpu().numpy()
        
        valid_period_mask = ~np.isnan(period_target)
        if np.any(valid_period_mask):
            period_mae = np.mean(np.abs(period_pred[valid_period_mask] - period_target[valid_period_mask]))
            metrics['peak_period_mae'] = period_mae
        
        # Energy conservation check
        total_energy_pred = pred_combined[:, 4].cpu().numpy()
        swell_energy_pred = pred_combined[:, 6].cpu().numpy()
        windsea_energy_pred = pred_combined[:, 7].cpu().numpy()
        
        energy_conservation_error = np.mean(np.abs(total_energy_pred - (swell_energy_pred + windsea_energy_pred)))
        metrics['energy_conservation_error'] = energy_conservation_error
        
        # Fraction constraint check
        swell_fraction_pred = pred_combined[:, 8].cpu().numpy()
        windsea_fraction_pred = pred_combined[:, 9].cpu().numpy()
        fraction_sum_error = np.mean(np.abs((swell_fraction_pred + windsea_fraction_pred) - 1.0))
        metrics['fraction_constraint_error'] = fraction_sum_error
        
        return metrics
    
    def _compute_detailed_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute detailed per-feature validation metrics"""
        metrics = {}
        
        for i, feature_name in enumerate(self.feature_names):
            pred_feature = predictions[:, i]
            target_feature = targets[:, i]
            
            # Only compute metrics where we have valid targets
            valid_mask = ~np.isnan(target_feature)
            if not np.any(valid_mask):
                continue
            
            pred_valid = pred_feature[valid_mask]
            target_valid = target_feature[valid_mask]
            
            # Mean Absolute Error
            mae = np.mean(np.abs(pred_valid - target_valid))
            metrics[f'{feature_name}_mae'] = mae
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean((pred_valid - target_valid)**2))
            metrics[f'{feature_name}_rmse'] = rmse
            
            # Correlation coefficient
            if len(target_valid) > 1 and np.std(target_valid) > 0 and np.std(pred_valid) > 0:
                corr = np.corrcoef(pred_valid, target_valid)[0, 1]
                metrics[f'{feature_name}_correlation'] = corr
        
        return metrics
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
              val_loader: torch.utils.data.DataLoader, 
              num_epochs: int = 100) -> Dict[str, List[float]]:
        """Complete training loop with early stopping and checkpointing"""
        
        print(f"🌊 Starting Physics-Informed Wave Forecasting Training")
        print(f"📊 Features: {len(self.feature_names)} physics-informed dimensions")
        print(f"🎯 Epochs: {num_epochs}, Device: {self.device}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            epoch_start = datetime.now()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Track metrics
            all_metrics = {**train_metrics, **val_metrics}
            for metric_name, metric_value in all_metrics.items():
                self.metrics_history[metric_name].append(metric_value)
            
            # Log to wandb
            if self.log_wandb:
                wandb.log(all_metrics, step=epoch)
            
            # Early stopping check
            val_loss = val_metrics['val_total_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, 'best_model.pth')
            else:
                self.patience_counter += 1
            
            # Print epoch summary
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_metrics['train_total_loss']:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Wave Ht MAE: {val_metrics.get('val_wave_height_mae', 0):.3f}m | "
                  f"Time: {epoch_time:.1f}s")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping
            if self.patience_counter >= self.max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("🎉 Training completed!")
        return dict(self.metrics_history)
    
    def _save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics_history': dict(self.metrics_history),
            'feature_names': self.feature_names
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves for analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot main losses
        plot_configs = [
            ('total_loss', 'Total Loss'),
            ('spectral_loss', 'Spectral Physics Loss'),
            ('directional_loss', 'Directional Physics Loss'),
            ('wave_height_mae', 'Wave Height MAE (m)'),
            ('energy_conservation_error', 'Energy Conservation Error'),
            ('fraction_constraint_error', 'Fraction Constraint Error')
        ]
        
        for i, (metric_key, title) in enumerate(plot_configs):
            ax = axes[i]
            
            train_key = f'train_{metric_key}'
            val_key = f'val_{metric_key}'
            
            if train_key in self.metrics_history:
                ax.plot(self.metrics_history[train_key], label='Train', alpha=0.8)
            if val_key in self.metrics_history:
                ax.plot(self.metrics_history[val_key], label='Validation', alpha=0.8)
            
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def main_training_pipeline(features_file: str = "pacific_enhanced_features.json"):
    """Complete training pipeline for your enhanced features"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create enhanced dataloaders
    from enhanced_wave_sequencer import create_enhanced_dataloaders
    
    train_loader, val_loader = create_enhanced_dataloaders(
        features_file=features_file,
        sequence_length=24,  # 24 hours of history
        batch_size=8,       # Start small for your data size
        split_ratio=0.8,
        min_stations=3,     # Require at least 3 stations per sequence
        normalize_features=True,
        interpolate_missing=True
    )
    
    print(f"📊 Created dataloaders: {len(train_loader)} train, {len(val_loader)} val batches")
    
    # Create physics-informed model
    from physics_wave_transformer import create_physics_model
    
    model, criterion, optimizer, scheduler = create_physics_model(device=device)
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = PhysicsWaveTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_wandb=True
    )
    
    # Train the model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100
    )
    
    # Plot training curves
    trainer.plot_training_curves('training_curves.png')
    
    # Save final metrics
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("🌊 Physics-informed wave forecasting training complete!")
    print("📊 Check 'training_curves.png' for training progress")
    print("💾 Best model saved as 'best_model.pth'")


if __name__ == "__main__":
    main_training_pipeline()