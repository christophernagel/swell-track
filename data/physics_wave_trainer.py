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
import time
import pickle
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class PhysicsWaveTrainer:
    """
    Enhanced trainer for physics-informed wave forecasting with detailed logging.
    """
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], device: str,
                 feature_names: List[str], log_wandb: bool = True, save_dir: str = 'wave_model_checkpoints'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.feature_names = feature_names
        self.log_wandb = log_wandb
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.metrics_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 15  # Early stopping patience

        if self.log_wandb:
            wandb.init(project="physics-wave-forecasting", config={
                "model_type": self.model.__class__.__name__,
                "features": f"{len(feature_names)}D_enhanced_physics"
            })
            wandb.watch(self.model, log_freq=100)

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch with detailed logging"""
        self.model.train()
        epoch_metrics = defaultdict(list)
        batch_times = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            sequence_features = batch['sequence_features'].to(self.device)
            target_features = batch['target_features'].to(self.device)
            station_coords = batch['station_coords'].to(self.device)
            station_masks = batch['station_masks'].to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(
                sequence_features=sequence_features,
                station_coords=station_coords,
                station_mask=station_masks
            )
            
            losses = self.criterion(predictions, target_features, station_masks)
            total_loss = losses['total']
            
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            for loss_name, loss_value in losses.items():
                epoch_metrics[f'train_{loss_name}_loss'].append(loss_value.item())
            
            with torch.no_grad():
                physics_metrics = self._compute_physics_metrics(predictions, target_features, station_masks)
            for metric_name, metric_value in physics_metrics.items():
                epoch_metrics[f'train_{metric_name}'].append(metric_value)
            
            # Detailed batch logging
            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                progress = (batch_idx + 1) / len(train_loader) * 100
                avg_batch_time = np.mean(batch_times[-10:])
                eta_seconds = avg_batch_time * (len(train_loader) - batch_idx - 1)
                
                print(f"\rBatch {batch_idx+1:3d}/{len(train_loader)} ({progress:5.1f}%) | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Wave Ht MAE: {physics_metrics.get('wave_height_mae', 0):.3f}m | "
                      f"Grad: {grad_norm:.2f} | "
                      f"Time: {batch_time:.2f}s | "
                      f"ETA: {eta_seconds/60:.1f}m", end="")
        print() # Newline after epoch progress bar
        return {k: np.mean(v) for k, v in epoch_metrics.items()}

    def validate_epoch(self, val_loader: torch.utils.data.DataLoader, is_test: bool = False) -> Dict[str, float]:
        """Validate or test for one epoch with detailed logging"""
        self.model.eval()
        epoch_metrics = defaultdict(list)
        all_predictions, all_targets = [], []
        prefix = "test" if is_test else "val"
        
        with torch.no_grad():
            for batch in val_loader:
                sequence_features = batch['sequence_features'].to(self.device)
                target_features = batch['target_features'].to(self.device)
                station_coords = batch['station_coords'].to(self.device)
                station_masks = batch['station_masks'].to(self.device)

                predictions = self.model(
                    sequence_features=sequence_features,
                    station_coords=station_coords,
                    station_mask=station_masks
                )
                
                losses = self.criterion(predictions, target_features, station_masks)
                
                for loss_name, loss_value in losses.items():
                    epoch_metrics[f'{prefix}_{loss_name}_loss'].append(loss_value.item())
                
                physics_metrics = self._compute_physics_metrics(predictions, target_features, station_masks)
                for metric_name, metric_value in physics_metrics.items():
                    epoch_metrics[f'{prefix}_{metric_name}'].append(metric_value)

                all_predictions.append(predictions['combined'][station_masks].cpu().numpy())
                all_targets.append(target_features[station_masks].cpu().numpy())
        
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        if all_predictions:
            all_predictions = np.vstack(all_predictions)
            all_targets = np.vstack(all_targets)
            detailed_metrics = self._compute_detailed_metrics(all_predictions, all_targets)
            avg_metrics.update({f'{prefix}_{k}': v for k, v in detailed_metrics.items()})
            
        return avg_metrics

    def _compute_physics_metrics(self, predictions: Dict, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Computes key physics-based performance metrics."""
        metrics = {}
        pred = predictions['combined'][mask]
        targ = targets[mask]
        if pred.shape[0] == 0: return metrics

        # Wave Height MAE
        metrics['wave_height_mae'] = torch.abs(pred[:, 0] - targ[:, 0]).mean().item()
        # Peak Period MAE
        metrics['peak_period_mae'] = torch.abs(pred[:, 1] - targ[:, 1]).mean().item()
        # Energy Conservation Error
        energy_sum = pred[:, 6] + pred[:, 7] # swell + windsea
        metrics['energy_conservation_error'] = torch.abs(pred[:, 4] - energy_sum).mean().item()
        # Fraction Constraint Error
        fraction_sum = pred[:, 8] + pred[:, 9]
        metrics['fraction_constraint_error'] = torch.abs(fraction_sum - 1.0).mean().item()
        
        return metrics

    def _compute_detailed_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Computes detailed per-feature validation metrics from numpy arrays."""
        metrics = {}
        for i, name in enumerate(self.feature_names):
            pred_feat, targ_feat = predictions[:, i], targets[:, i]
            valid_mask = ~np.isnan(targ_feat)
            if not np.any(valid_mask): continue
            
            pred_valid, targ_valid = pred_feat[valid_mask], targ_feat[valid_mask]
            metrics[f'{name}_mae'] = np.mean(np.abs(pred_valid - targ_valid))
            if np.std(targ_valid) > 0 and np.std(pred_valid) > 0:
                metrics[f'{name}_correlation'] = np.corrcoef(pred_valid, targ_valid)[0, 1]
        return metrics

    def train(self, train_loader, val_loader, test_loader, num_epochs, normalizer):
        """Main training loop with enhanced logging and testing."""
        print(f"🏗️  Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            print("\n" + "="*80)
            print(f"🚀 EPOCH {epoch+1}/{num_epochs} | ⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
            print("="*80)

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_total_loss'])
                else: self.scheduler.step()
            
            all_metrics = {**train_metrics, **val_metrics}
            for k, v in all_metrics.items(): self.metrics_history[k].append(v)
            if self.log_wandb: wandb.log(all_metrics, step=epoch)

            # --- Epoch Summary & Checkpointing ---
            epoch_time = time.time() - epoch_start
            print(f"\n🏁 EPOCH {epoch+1} SUMMARY | ⏱️  Time: {epoch_time:.1f}s")
            print(f"  - Train Loss: {all_metrics['train_total_loss']:.4f} | Val Loss: {all_metrics['val_total_loss']:.4f}")
            print(f"  - Val Wave Ht MAE: {all_metrics.get('val_wave_height_mae', 0):.3f}m | Val Energy Cons: {all_metrics.get('val_energy_conservation_error', 0):.4f}")
            
            improved = all_metrics['val_total_loss'] < self.best_val_loss
            if improved:
                self.best_val_loss = all_metrics['val_total_loss']
                self.patience_counter = 0
                self._save_checkpoint(epoch, 'best_model.pth', normalizer)
                print(f"  - ✅ Val loss improved to {self.best_val_loss:.4f}. Saving best model.")
            else:
                self.patience_counter += 1
                print(f"  - ⏳ No improvement for {self.patience_counter}/{self.max_patience} epochs.")

            if (epoch + 1) % 10 == 0: self._save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth', normalizer)
            if self.patience_counter >= self.max_patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        print("\n" + "="*80)
        print("🎉 Training loop completed!")
        
        # --- Final Test Evaluation ---
        print("🧪 Evaluating best model on the held-out test set...")
        
        # CRITICAL FIX: Add weights_only=False for PyTorch 2.6 compatibility
        checkpoint = torch.load(self.save_dir / 'best_model.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.validate_epoch(test_loader, is_test=True)
        self.metrics_history['test_metrics'] = test_metrics

        return dict(self.metrics_history)

    def _save_checkpoint(self, epoch: int, filename: str, normalizer):
        """Saves model checkpoint and the data normalizer."""
        checkpoint_path = self.save_dir / filename
        normalizer_path = self.save_dir / 'wave_normalizer.pkl'

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        if self.scheduler: checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, checkpoint_path)

        with open(normalizer_path, 'wb') as f:
            pickle.dump(normalizer, f)

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plots training and validation curves for key metrics."""
        sns.set_theme(style="whitegrid")
        plot_configs = [
            ('total_loss', 'Total Loss'), ('spectral_loss', 'Spectral Physics Loss'),
            ('directional_loss', 'Directional Physics Loss'), ('wave_height_mae', 'Wave Height MAE (m)'),
            ('energy_conservation_error', 'Energy Conservation Error'), ('fraction_constraint_error', 'Fraction Constraint Error')
        ]
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle('Training & Validation Performance', fontsize=16)
        axes = axes.flatten()

        for i, (metric, title) in enumerate(plot_configs):
            ax = axes[i]
            if f'train_{metric}' in self.metrics_history:
                ax.plot(self.metrics_history[f'train_{metric}'], label='Train', alpha=0.8)
            if f'val_{metric}' in self.metrics_history:
                ax.plot(self.metrics_history[f'val_{metric}'], label='Validation', alpha=0.9, linestyle='--')
            ax.set_title(title)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
            ax.legend(); ax.grid(True, alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()