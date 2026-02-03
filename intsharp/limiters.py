"""
Slope limiters for higher-order reconstruction.

Implements various slope limiters for MUSCL-type schemes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def minmod(a: NDArray, b: NDArray) -> NDArray:
    """
    Minmod function: returns the value with smaller magnitude if same sign, else 0.
    """
    return np.where(
        a * b > 0,
        np.sign(a) * np.minimum(np.abs(a), np.abs(b)),
        0.0
    )


def barth_jespersen_1d(
    q: NDArray[np.float64],
    q_min_neighbors: NDArray[np.float64],
    q_max_neighbors: NDArray[np.float64],
    delta_q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Barth-Jespersen limiter for 1D.
    
    Computes limiter φ ∈ [0, 1] such that the reconstructed values
    q ± 0.5 * φ * delta_q stay within [q_min_neighbors, q_max_neighbors].
    
    Parameters
    ----------
    q : NDArray
        Cell-centered values.
    q_min_neighbors : NDArray
        Minimum of cell value and its neighbors.
    q_max_neighbors : NDArray
        Maximum of cell value and its neighbors.
    delta_q : NDArray
        Gradient (difference) to be limited.
    
    Returns
    -------
    NDArray
        Limiter values φ ∈ [0, 1].
    """
    phi = np.ones_like(q)
    
    # Small tolerance for numerical stability
    eps = 1e-12
    
    # For each face of the cell, compute the limiter
    # In 1D, we have left and right faces at ±0.5*delta_q
    
    for sign in [-0.5, 0.5]:
        q_face = q + sign * delta_q
        
        # Where face value exceeds maximum
        mask_high = q_face > q_max_neighbors + eps
        phi_high = np.where(
            mask_high & (np.abs(delta_q) > eps),
            (q_max_neighbors - q) / (sign * delta_q + eps * np.sign(sign * delta_q)),
            1.0
        )
        
        # Where face value is below minimum
        mask_low = q_face < q_min_neighbors - eps
        phi_low = np.where(
            mask_low & (np.abs(delta_q) > eps),
            (q_min_neighbors - q) / (sign * delta_q + eps * np.sign(sign * delta_q)),
            1.0
        )
        
        # Take minimum of all constraints
        phi = np.minimum(phi, np.maximum(phi_high, 0.0))
        phi = np.minimum(phi, np.maximum(phi_low, 0.0))
    
    return np.clip(phi, 0.0, 1.0)


def compute_limited_gradients_1d(
    q_ext: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute Barth-Jespersen limited gradients for 1D array.
    
    Parameters
    ----------
    q_ext : NDArray
        Extended array with ghost cells: [ghost_L, q_0, q_1, ..., q_{n-1}, ghost_R]
        Length is n + 2.
    
    Returns
    -------
    NDArray
        Limited gradients for interior cells (length n).
    """
    n = len(q_ext) - 2  # Number of interior cells
    
    # Cell values (interior)
    q = q_ext[1:-1]
    
    # Neighbors
    q_left = q_ext[:-2]   # Left neighbors
    q_right = q_ext[2:]   # Right neighbors
    
    # Compute min/max of cell and its neighbors
    q_min = np.minimum(np.minimum(q, q_left), q_right)
    q_max = np.maximum(np.maximum(q, q_left), q_right)
    
    # Centered gradient estimate
    delta_q = 0.5 * (q_right - q_left)
    
    # Apply Barth-Jespersen limiter
    phi = barth_jespersen_1d(q, q_min, q_max, delta_q)
    
    return phi * delta_q


def muscl_reconstruct_1d(
    q_ext: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    MUSCL reconstruction with Barth-Jespersen limiter.
    
    Computes left and right states at cell interfaces.
    
    Parameters
    ----------
    q_ext : NDArray
        Extended array with ghost cells: [ghost_L, q_0, ..., q_{n-1}, ghost_R]
        Length is n + 2.
    
    Returns
    -------
    tuple[NDArray, NDArray]
        (q_L, q_R) at interfaces. Each has length n + 1 (interfaces 0 to n).
        q_L[i] is the left state at interface i (from cell i-1).
        q_R[i] is the right state at interface i (from cell i).
    """
    # Compute limited gradients for all cells including ghosts
    # For proper reconstruction, we need gradients at ghost cells too
    # So we extend further for gradient computation
    
    n = len(q_ext) - 2  # Interior cells
    
    # For gradient at cell i, we need cells i-1, i, i+1
    # Ghost cells are at index 0 and n+1
    # We'll use one-sided differences at boundaries
    
    # Compute gradients for interior cells (indices 1 to n in q_ext)
    # Using centered differences with limiting
    
    delta_q = np.zeros_like(q_ext)
    
    # Interior cells: use centered difference
    for i in range(1, len(q_ext) - 1):
        q_left = q_ext[i - 1]
        q_center = q_ext[i]
        q_right = q_ext[i + 1]
        
        # Centered gradient
        grad_centered = 0.5 * (q_right - q_left)
        
        # Min/max of neighbors
        q_min = min(q_left, q_center, q_right)
        q_max = max(q_left, q_center, q_right)
        
        # Limit so that q_center ± 0.5*grad stays in [q_min, q_max]
        phi = 1.0
        eps = 1e-12
        
        for sign in [-0.5, 0.5]:
            q_face = q_center + sign * grad_centered
            if q_face > q_max + eps and abs(grad_centered) > eps:
                phi = min(phi, max(0, (q_max - q_center) / (sign * grad_centered)))
            if q_face < q_min - eps and abs(grad_centered) > eps:
                phi = min(phi, max(0, (q_min - q_center) / (sign * grad_centered)))
        
        delta_q[i] = phi * grad_centered
    
    # Reconstruct at interfaces
    # Interface i is between cells i-1 and i (in extended indexing: i and i+1)
    # q_L at interface i comes from cell i-1 (extended index i), extrapolated right
    # q_R at interface i comes from cell i (extended index i+1), extrapolated left
    
    # Number of interfaces: n + 1 (between ghost and first interior, 
    # between each pair of interior cells, between last interior and ghost)
    
    q_L = np.zeros(n + 1)
    q_R = np.zeros(n + 1)
    
    for i in range(n + 1):
        # Left state: from cell at extended index i, extrapolated to right face
        q_L[i] = q_ext[i] + 0.5 * delta_q[i]
        # Right state: from cell at extended index i+1, extrapolated to left face
        q_R[i] = q_ext[i + 1] - 0.5 * delta_q[i + 1]
    
    return q_L, q_R
