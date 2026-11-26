"""Seed management module for reproducible execution across random generators."""

import random
import os
import numpy as np

def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python random, NumPy, environment, and PyTorch (if installed).
    
    Args:
        seed: The integer seed value to apply.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
