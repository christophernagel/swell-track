"""
model_trainer.py
----------------
Defines a WaveModelTrainer class that handles multi-task training,
loss computation, and validation for the EnhancedWaveTransformer.
"""

import torch
import numpy as np
from typing import Dict, Optional
from collections import defaultdict


class WaveModelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        task_weights: Optional[Dict[str, float]] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        
        # Default task weights
        self.task_weights = task_weights or {
            'buoy_states': 1.0,
            'network_state': 1.0,
            'surf_conditions': 1.0
        }
        
        # Define task-specific loss functions
        self.loss_fns = {
            'buoy_states': self._buoy_prediction_loss,
            'network_state': self._network_state_loss,
            'surf_conditions': self._surf_conditions_loss
        }
        
        self.metrics_history = defaultdict(list)
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        validation_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        for batch in dataloader:
            batch_metrics = self._train_step(batch)
            for k, v in batch_metrics.items():
                epoch_metrics[k].append(v)
        
        # Compute average metrics
        metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        if validation_loader:
            val_metrics = self.validate(validation_loader)
            for k, v in val_metrics.items():
                metrics[f'val_{k}'] = v
        
        for k, v in metrics.items():
            self.metrics_history[k].append(v)
        
        return metrics
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad()

        # Move data to device
        device_batch = {}
        for k, v in batch.items():
            if k == 'timestamps':  # assume timestamps is a list of datetime objects
                device_batch[k] = v
            else:
                device_batch[k] = v.to(self.device)
        
        outputs = self.model(
            x=device_batch['features'],
            timestamps=device_batch['timestamps'],
            mask=device_batch.get('mask'),
            missing_buoys=device_batch.get('missing_buoys')
        )
        
        # Compute task losses
        losses = {}
        metrics = {}
        for task, loss_fn in self.loss_fns.items():
            target_key = f'{task}_target'
            if task in outputs and target_key in device_batch:
                task_loss = loss_fn(
                    outputs[task],
                    device_batch[target_key],
                    device_batch.get('missing_buoys')
                )
                weighted_loss = task_loss * self.task_weights[task]
                losses[task] = weighted_loss
                metrics[f'{task}_loss'] = task_loss.item()
        
        total_loss = sum(losses.values()) if len(losses) > 0 else 0
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return metrics
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in dataloader:
                device_batch = {}
                for k, v in batch.items():
                    if k == 'timestamps':
                        device_batch[k] = v
                    else:
                        device_batch[k] = v.to(self.device)
                
                outputs = self.model(
                    x=device_batch['features'],
                    timestamps=device_batch['timestamps'],
                    mask=device_batch.get('mask'),
                    missing_buoys=device_batch.get('missing_buoys')
                )
                
                # Compute validation losses
                for task, loss_fn in self.loss_fns.items():
                    target_key = f'{task}_target'
                    if task in outputs and target_key in device_batch:
                        loss_val = loss_fn(
                            outputs[task],
                            device_batch[target_key],
                            device_batch.get('missing_buoys')
                        )
                        val_metrics[f'{task}_loss'].append(loss_val.item())
                        
                        # Additional metrics if needed (e.g., accuracy)
                        if task == 'surf_conditions':
                            acc = self._compute_surf_accuracy(outputs[task], device_batch[target_key])
                            val_metrics[f'{task}_accuracy'].append(acc)
        
        return {k: np.mean(v) for k, v in val_metrics.items()}

    def _buoy_prediction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        missing_buoys: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Simple MSE with optional missing buoy masking
        mse = torch.nn.functional.mse_loss(pred, target, reduction='none')
        if missing_buoys is not None:
            # Assume shape: (batch_size, num_buoys, feature_dim)
            # Expand missing to match feature dim
            # For simplicity, missing_buoys might only have shape (batch_size, num_buoys)
            pass
        return mse.mean()

    def _network_state_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        missing_buoys: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Typically a single vector (batch_size, 1, d_model) or (batch_size, 1, some_dim)
        return torch.nn.functional.mse_loss(pred, target)

    def _surf_conditions_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        missing_buoys: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Example: MSE for wave height + Cross-Entropy for quality categories.
        Here we assume pred is split: first 1 is continuous, next 3 are categorical.
        """
        height_pred, quality_pred = pred.split([1, 3], dim=-1)
        height_tgt, quality_tgt = target.split([1, 3], dim=-1)

        height_loss = torch.nn.functional.mse_loss(height_pred, height_tgt)
        quality_loss = torch.nn.functional.cross_entropy(
            quality_pred.view(-1, 3),
            quality_tgt.view(-1).long()
        )
        return height_loss + quality_loss

    @staticmethod
    def _compute_surf_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        _, quality_pred = pred.split([1, 3], dim=-1)
        _, quality_tgt = target.split([1, 3], dim=-1)
        pred_classes = torch.argmax(quality_pred, dim=-1)
        correct = (pred_classes == quality_tgt.long()).float().mean().item()
        return correct
