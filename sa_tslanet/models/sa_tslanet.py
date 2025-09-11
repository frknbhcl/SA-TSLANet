"""SA-TSLANet model implementation."""

import torch
import torch.nn as nn
from einops import rearrange
from .components import Modified_TSLANet_layer


class TSLANet_Detect(nn.Module):
    """SA-TSLANet Detection Model for chatter identification."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size_detect
        self.stride = self.patch_size // 2
        
        # Calculate number of patches
        num_patches = int(((config.seq_len + config.pred_len) - self.patch_size) / self.stride + 1)
        
        # Layers
        self.input_layer = nn.Linear(self.patch_size, config.emb_dim)
        
        # Create stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, config.dropout, config.depth)]
        
        # TSLANet blocks
        self.tsla_blocks = nn.ModuleList([
            Modified_TSLANet_layer(
                dim=config.emb_dim,
                drop=config.dropout,
                drop_path=dpr[i],
                stride=self.stride,
                adaptive_filter=config.adaptive_filter,
                use_icb=config.ICB,
                use_asb=config.ASB
            )
            for i in range(config.depth)
        ])
        
        # Output layer
        self.out_layer = nn.Linear(
            config.emb_dim * num_patches,
            config.seq_len + config.pred_len
        )

    def adjust_spindle_freq(self, spindle_freq, M):
        """Adjust spindle frequencies for rearranged batch dimension."""
        return spindle_freq.repeat_interleave(M)

    def forward(self, x, spindle_freq, masks):
        """
        Forward pass of SA-TSLANet.
        
        Args:
            x: Input tensor [B, L, M] 
            spindle_freq: Spindle frequencies [B]
            masks: Frequency masks [B, freq_bins]
            
        Returns:
            outputs: Reconstructed signal
            x_weighted_high: High frequency components
            x_weighted_spindle: Spindle components
            x_weighted_chatter: Chatter components
        """
        B, L, M = x.shape
        
        # Normalize input
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev
        
        # Adjust spindle frequencies for new batch dimension
        adjusted_spindle_freq = self.adjust_spindle_freq(spindle_freq, M)
        
        # Rearrange and create patches
        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        
        # Patch embedding
        x = self.input_layer(x)
        
        # Process through TSLANet blocks
        for tsla_blk in self.tsla_blocks:
            x, x_weighted_high, x_weighted_spindle, x_weighted_chatter = tsla_blk(
                x, adjusted_spindle_freq, masks
            )
        
        # Output projection
        outputs = self.out_layer(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        
        # Denormalize
        outputs = outputs * stdev
        outputs = outputs + means
        
        return outputs, x_weighted_high, x_weighted_spindle, x_weighted_chatter
