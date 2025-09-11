"""Training utilities for SA-TSLANet."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple


class ChatterDetectionTrainer:
    """Trainer for SA-TSLANet chatter detection model."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizers
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=2,
            verbose=True
        )
        
        # History tracking
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': []
        }

    def calculate_spectral_losses(self, x_global, x_spindle, x_chatter, masks):
        """Calculate spectral component losses."""
        # Convert to magnitude in frequency domain
        x_global_mag = torch.abs(x_global)**2
        x_spindle_mag = torch.abs(x_spindle)**2
        x_chatter_mag = torch.abs(x_chatter)**2
        
        # Create masks
        spindle_mask = masks.unsqueeze(-1)
        chatter_mask = 1 - spindle_mask
        
        # Count active frequencies
        spindle_mask_sum = torch.sum(spindle_mask, dim=(1,2), keepdim=True) + 1e-8
        chatter_mask_sum = torch.sum(chatter_mask, dim=(1,2), keepdim=True) + 1e-8
        
        # Normalize magnitudes
        x_global_mag_norm = x_global_mag / torch.sum(x_global_mag, dim=(1,2), keepdim=True)
        x_spindle_mag_norm = x_spindle_mag / torch.sum(x_spindle_mag, dim=(1,2), keepdim=True)
        
        # Global-spindle similarity loss
        global_spindle_loss = torch.mean((x_global_mag_norm - x_spindle_mag_norm)**2)
        
        # Energy ratio loss
        spindle_energy = torch.sum(x_spindle_mag, dim=(1,2), keepdim=True)
        chatter_energy = torch.sum(x_chatter_mag, dim=(1,2), keepdim=True)
        
        ratio_loss = torch.mean(chatter_energy / (spindle_energy + 1e-8))
        
        return global_spindle_loss, ratio_loss

    def crest_factor_loss(self, reconstruction, target, dim=1):
        """Calculate crest factor loss."""
        # Peak values
        recon_peak = torch.max(torch.abs(reconstruction), dim=dim, keepdim=True)[0]
        target_peak = torch.max(torch.abs(target), dim=dim, keepdim=True)[0]
        
        # RMS values
        recon_rms = torch.sqrt(torch.mean(reconstruction ** 2, dim=dim, keepdim=True) + 1e-8)
        target_rms = torch.sqrt(torch.mean(target ** 2, dim=dim, keepdim=True) + 1e-8)
        
        # Crest factors
        recon_crest = recon_peak / (recon_rms + 1e-8)
        target_crest = target_peak / (target_rms + 1e-8)
        
        # Loss
        crest_loss = torch.mean((recon_crest - target_crest) ** 2)
        
        return crest_loss

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch in progress_bar:
            # Move to device
            batch_data, labels, sf, exp_idx, masks = batch
            batch_data = batch_data.float().to(self.device)
            sf = sf.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            reconstruction, x_global, x_spindle, x_chatter = self.model(batch_data, sf, masks)
            
            # Calculate losses
            crest_loss = self.crest_factor_loss(reconstruction, batch_data)
            global_spindle_loss, ratio_loss = self.calculate_spectral_losses(
                x_global, x_spindle, x_chatter, masks
            )
            
            # Weighted total loss
            total_loss = (
                self.config.a_crest * crest_loss +
                self.config.a_spindle * global_spindle_loss +
                self.config.a_ratio * ratio_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()
            
            # Track loss
            epoch_losses.append(total_loss.item())
            progress_bar.set_postfix({'loss': total_loss.item()})
        
        return np.mean(epoch_losses)

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                batch_data, labels, sf, exp_idx, masks = batch
                batch_data = batch_data.float().to(self.device)
                sf = sf.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                reconstruction, x_global, x_spindle, x_chatter = self.model(batch_data, sf, masks)
                
                # Calculate losses
                crest_loss = self.crest_factor_loss(reconstruction, batch_data)
                global_spindle_loss, ratio_loss = self.calculate_spectral_losses(
                    x_global, x_spindle, x_chatter, masks
                )
                
                # Total loss
                total_loss = (
                    self.config.a_crest * crest_loss +
                    self.config.a_spindle * global_spindle_loss +
                    self.config.a_ratio * ratio_loss
                )
                
                val_losses.append(total_loss.item())
        
        return np.mean(val_losses)

    def fit(self, train_loader, val_loader, epochs):
        """Train the model for multiple epochs."""
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_losses'].append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.history['val_losses'].append(val_loss)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return self.model, self.history

    def save_checkpoint(self, path, epoch, best_loss):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': best_loss,
            'config': self.config,
            'history': self.history
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch'], checkpoint['best_loss']
