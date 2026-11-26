"""Smoke tests for package initialization and environment sanity."""

def test_imports():
    import ml.utils.seed
    import ml.utils.io
    import ml.utils.hashing
    import ml.utils.logging_utils
    import ml.utils.paths
    import ml.utils.config
    assert True

def test_seed_execution():
    from ml.utils.seed import set_seed
    import random
    import numpy as np
    
    set_seed(42)
    val1 = random.random()
    np_val1 = np.random.rand()
    
    set_seed(42)
    val2 = random.random()
    np_val2 = np.random.rand()
    
    assert val1 == val2
    assert np_val1 == np_val2
