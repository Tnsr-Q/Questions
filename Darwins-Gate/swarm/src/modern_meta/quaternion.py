import numpy as np
from scipy.spatial.transform import Rotation

def quaternion_to_matrix(q, normalize=True):
    """Robust quaternion to rotation matrix conversion."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must be length-4, got shape {q.shape}")
    
    if normalize:
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            # Fallback to identity quaternion if zero vector provided
            return np.eye(3)
        q = q / norm
    
    return Rotation.from_quat(q).as_matrix()

def random_unit_quaternion(seed=None):
    """
    Generate uniform random unit quaternion (Shoemake's algorithm).
    Ensures uniform distribution over S^3.
    """
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random(3)
    return np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])

def generate_geometric_projections(n_proj=5, rank=2, seed=None):
    """
    Generate rank-k projection matrices in 3D space.
    Returns dense matrices (3x3 is too small to benefit from sparse format).
    """
    rng = np.random.default_rng(seed)
    projections = []
    
    for _ in range(n_proj):
        q = random_unit_quaternion(seed=rng.integers(0, 10**9))
        R = quaternion_to_matrix(q, normalize=True)
        # Select first 'rank' columns to define the subspace
        basis = R[:, :rank]
        # P = Basis * Basis.T (Projection onto the subspace)
        P = basis @ basis.T
        projections.append(P) # Keep dense for 3x3
    
    return projections
