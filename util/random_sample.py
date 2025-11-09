"""A module to sample randomly from various distributions."""
import numpy as np
import random

def sample_uniform(min_val, max_val, num_samples):
    """Sample from uniform distribution."""
  
    """Samples from a uniform distribution.
    
    Samples num_samples integer values from the range [min, max)
    uniformly at random.
    
    Args:
        min_val (int): Minimum value (inclusive).
        max_val (int): Maximum value (exclusive).
        num_samples (int): Number of samples to be drawn.
    
    Returns:
        A list of randomly sampled values.
    """
     if num_samples == 1:
        return random.randint(min_val, max_val - 1)
    return [random.randint(min_val, max_val - 1) for _ in range(num_samples)]

def sample_triangle(num_samples):
    """Samples from a discrete triangle distribution.
    
    Samples num_samples values from [-1, 0, 1] with probabilities
    [0.25, 0.5, 0.25], respectively.
    
    Args:
        num_samples (int): Number of samples to be drawn.
    
    Returns:
        A list of randomly sampled values.
    """
    r = np.random.randint(0, 4, size=num_samples)
    sample = np.where(r == 0, -1, np.where(r == 1, 1, 0))
    return sample.tolist()

def sample_hamming_weight_vector(length, hamming_weight):
    """Samples from a Hamming weight distribution.
    
    Samples uniformly from the set [-1, 0, 1] such that the
    resulting vector has exactly h nonzero values.
    
    Args:
        length (int): Length of resulting vector.
        hamming_weight (int): Hamming weight h of resulting vector.
    
    Returns:
        A list of randomly sampled values.
    """
    sample = np.zeros(length, dtype=np.int64)
    indices = np.random.choice(length, size=hamming_weight, replace=False)
    values = np.random.choice([-1, 1], size=hamming_weight)
    sample[indices] = values
    return sample.tolist()

def sample_random_complex_vector(length):
    """Samples a random complex vector,
    
    Samples a vector with elements of the form a + bi where a and b
    are chosen uniformly at random from the set [0, 1).
    
    Args:
        length (int): Length of vector.
    
    Returns:
        A list of randomly sampled complex values.
    """
    a = np.random.random(length)
    b = np.random.random(length)
    sample = a + b * 1j
    return sample.tolist()

def sample_random_real_vector(length):
    """Samples a random complex vector,
    
    Samples a vector with elements chosen uniformly at random from
    the set [0, 1).
    
    Args:
        length (int): Length of vector.
    
    Returns:
        A list of randomly sampled real values.
    """
    sample = np.random.random(length)
    return sample.tolist()
