"""Data preprocessing utilities for SA-TSLANet."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional


class TimeSeriesPreprocessor:
    """Preprocessor for time series data with windowing and normalization."""
    
    def __init__(self, window_size=1500, stride=750):
        self.window_size = window_size
        self.stride = stride

    def create_windows(self, time_series, timestamps=None):
        """Create overlapping windows from time series data."""
        if time_series.ndim == 1:
            time_series = time_series.reshape(-1, 1)
        
        n_samples, n_channels = time_series.shape
        n_windows = ((n_samples - self.window_size) // self.stride) + 1
        
        windows = []
        window_times = [] if timestamps is not None else None
        
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            if end_idx <= n_samples:
                window = time_series[start_idx:end_idx]
                windows.append(window)
                
                if timestamps is not None:
                    time_window = timestamps[start_idx:end_idx]
                    window_times.append(time_window)
        
        windows = np.array(windows)
        window_times = np.array(window_times) if timestamps is not None else None
        
        return windows, window_times

    def normalize_window(self, window):
        """Normalize a single window to [-1, 1] range."""
        min_vals = np.min(window, axis=0, keepdims=True)
        max_vals = np.max(window, axis=0, keepdims=True)
        
        # Handle constant signals
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1.0, range_vals)
        
        # Scale to [-1, 1]
        normalized_window = 2 * (window - min_vals) / range_vals - 1
        
        return normalized_window

    def process_dataset(self, data_list, time_list=None, normalize=True):
        """Process multiple time series with per-window normalization."""
        if not isinstance(data_list, list):
            data_list = [data_list]
        if time_list is not None and not isinstance(time_list, list):
            time_list = [time_list]
        
        all_windows = []
        all_times = []
        
        for i, data in enumerate(data_list):
            times = time_list[i] if time_list is not None else None
            windows, window_times = self.create_windows(data, times)
            
            if normalize:
                normalized_windows = np.array([
                    self.normalize_window(window) for window in windows
                ])
                windows = normalized_windows
            
            all_windows.append(windows)
            if window_times is not None:
                all_times.append(window_times)
        
        return all_windows, all_times if time_list is not None else None


class ChatterDataset(Dataset):
    """Dataset for chatter detection."""
    
    def __init__(self, windows, labels, spindle_freqs, exp_ids, masks):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)
        self.spindle_freqs = torch.FloatTensor(spindle_freqs)
        self.exp_ids = torch.LongTensor(exp_ids)
        self.masks = masks

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            self.windows[idx],
            self.labels[idx],
            self.spindle_freqs[idx],
            self.exp_ids[idx],
            self.masks[idx]
        )
