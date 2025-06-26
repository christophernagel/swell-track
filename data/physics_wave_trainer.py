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
                    "criterion": str(criterion),
                    "optimizer": str(optimizer)
                }
            )
            wandb.watch(self.model, log="all")
    
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
                station_mask=station_masks
            )
            
            # Compute physics-informed loss
            losses = self.criterion(predictions, target_features, station_masks)
            total_loss = losses['total']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            with torch.no_grad():
                # Track loss metrics
                for loss_name, loss_value in losses.items():
                    epoch_metrics[f'train_{loss_name}'].append(loss_value.item())
                
                # Physics validation metrics
                physics_metrics = self._compute_physics_metrics(predictions, target_features, station_masks)
                for metric_name, metric_value in physics_metrics.items():
                    epoch_metrics[f'train_{metric_name}'].append(metric_value)
            
            # Log batch metrics
            if batch_idx % 20 == 0:
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
                
                predictions = self.model(
                    sequence_features=sequence_features,
                    station_coords=station_coords,
                    station_mask=station_masks
                )
                
                # Compute loss
                losses = self.criterion(predictions, target_features, station_masks)
                
                # Track metrics
                for loss_name, loss_value in losses.items():
                    epoch_metrics[f'val_{loss_name}'].append(loss_value.item())
                
                # Physics validation metrics
                physics_metrics = self._compute_physics_metrics(predictions, target_features, station_masks)
                for metric_name, metric_value in physics_metrics.items():
                    epoch_metrics[f'val_{metric_name}'].append(metric_value)
                
                # Collect predictions for detailed analysis
                if station_masks.any():
                    valid_preds = predictions['combined'][station_masks].cpu().numpy()
                    valid_targets = target_features[station_masks].cpu().numpy()
                    all_predictions.append(valid_preds)
                    all_targets.append(valid_targets)
        
        # Compute detailed validation metrics
        if all_predictions and len(all_predictions[0]) > 0:
            all_predictions = np.vstack(all_predictions)
            all_targets = np.vstack(all_targets)
            detailed_metrics = self._compute_detailed_metrics(all_predictions, all_targets)
            for k, v in detailed_metrics.items():
                epoch_metrics[f'val_{k}'].append(v)

        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def _compute_physics_metrics(self, predictions: Dict[str, torch.Tensor], 
                                targets: torch.Tensor, station_mask: torch.Tensor) -> Dict[str, float]:
        """Compute physics-based validation metrics within a no_grad context."""
        metrics = {}
        
        valid_mask = station_mask.bool()
        if not valid_mask.any():
            return metrics

        pred_combined = predictions['combined'][valid_mask]
        valid_targets = targets[valid_mask]

        height_pred = pred_combined[:, 0].detach().cpu().numpy()
        height_target = valid_targets[:, 0].detach().cpu().numpy()
        metrics['wave_height_mae'] = np.mean(np.abs(height_pred - height_target))
        metrics['wave_height_rmse'] = np.sqrt(np.mean((height_pred - height_target)**2))
        
        period_pred = pred_combined[:, 1].detach().cpu().numpy()
        period_target = valid_targets[:, 1].detach().cpu().numpy()
        metrics['peak_period_mae'] = np.mean(np.abs(period_pred - period_target))
        
        total_energy_pred = pred_combined[:, 4].detach().cpu().numpy()
        swell_energy_pred = pred_combined[:, 6].detach().cpu().numpy()
        windsea_energy_pred = pred_combined[:, 7].detach().cpu().numpy()
        metrics['energy_conservation_error'] = np.mean(np.abs(total_energy_pred - (swell_energy_pred + windsea_energy_pred)))
        
        swell_fraction_pred = pred_combined[:, 8].detach().cpu().numpy()
        windsea_fraction_pred = pred_combined[:, 9].detach().cpu().numpy()
        metrics['fraction_constraint_error'] = np.mean(np.abs((swell_fraction_pred + windsea_fraction_pred) - 1.0))
        
        return metrics
    
    def _compute_detailed_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute detailed per-feature validation metrics"""
        metrics = {}
        for i, feature_name in enumerate(self.feature_names):
            pred_feature, target_feature = predictions[:, i], targets[:, i]
            metrics[f'{feature_name}_mae'] = np.mean(np.abs(pred_feature - target_feature))
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
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            
            # --- FIX: Use the correct key 'val_total' ---
            val_loss = val_metrics.get('val_total', float('inf'))

            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler:
                self.scheduler.step()
            
            all_metrics = {**train_metrics, **val_metrics}
            for metric_name, metric_value in all_metrics.items():
                self.metrics_history[metric_name].append(metric_value)
            
            if self.log_wandb:
                wandb.log(all_metrics, step=epoch)
            
            if val_loss < self.best_val_loss:
                print(f"✅ Validation loss improved to {val_loss:.4f}. Saving best model.")
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, 'best_model.pth')
            else:
                self.patience_counter += 1
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            # --- FIX: Use the correct key 'train_total' ---
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_metrics.get('train_total', 0):.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Wave Ht MAE: {val_metrics.get('val_wave_height_mae', 0):.3f}m | "
                  f"Time: {epoch_time:.1f}s")
            
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
            
            if self.patience_counter >= self.max_patience:
                print(f"🕰️ Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break
        
        print("🎉 Training completed!")
        if self.log_wandb:
            wandb.finish()
        return dict(self.metrics_history)
    
    def _save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics_history': dict(self.metrics_history),
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves for analysis"""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        # --- FIX: Use correct metric keys from loss function ---
        plot_configs = [
            ('total', 'Total Loss'),
            ('spectral_loss', 'Spectral Loss'),
            ('directional_loss', 'Directional Loss'),
            ('height_mae', 'Wave Height MAE (m)'),
            ('energy_conservation_error', 'Energy Conservation Error'),
            ('fraction_constraint_error', 'Fraction Constraint Error')
        ]
        
        for i, (metric_key, title) in enumerate(plot_configs):
            ax = axes[i]
            train_key = f'train_{metric_key}'
            val_key = f'val_{metric_key}'
            
            if train_key in self.metrics_history and self.metrics_history[train_key]:
                ax.plot(self.metrics_history[train_key], label='Train', alpha=0.8, color='royalblue')
            if val_key in self.metrics_history and self.metrics_history[val_key]:
                ax.plot(self.metrics_history[val_key], label='Validation', alpha=0.9, color='darkorange', linestyle='--')
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.legend()
        
        plt.tight_layout(pad=3.0)
        fig.suptitle('Physics-Informed Training Curves', fontsize=20, y=1.03)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()