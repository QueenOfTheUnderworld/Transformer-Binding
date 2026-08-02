"""Shared utilities for the coherence-training line of work."""

import numpy as np
import torch


def set_seed(seed):
    """Seed numpy and torch together. Call immediately before each model's
    construction+training when comparing methods, so weight init and batch
    order are identical across arms and only the method differs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
