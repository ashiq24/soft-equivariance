import numpy as np
import torch
import scipy

def canonicalize_schur(T, U):
    """
    Takes a real Schur decomposition (T, U) and enforces a canonical form
    for its 2x2 blocks using a swap operation.

    The canonical form ensures that for any 2x2 block [[a, b], [-b, a]],
    the value of b is always non-negative.

    Args:
        T (np.ndarray): The Schur form matrix from a real Schur decomposition.
        U (np.ndarray): The corresponding orthogonal basis matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - T_canon (np.ndarray): The canonical Schur form matrix.
            - U_canon (np.ndarray): The corresponding transformed basis matrix.
    """
    # Create copies to modify
    T_canon = T.copy()
    U_canon = U.copy()
    
    n = T.shape[0]
    i = 0
    # swap matrix
    P = np.array([[0., 1.], [1., 0.]])
    while i < n - 1:
        # Check for a 2x2 block by looking at the sub-diagonal element
        if abs(T_canon[i + 1, i]) > 1e-8:
            block = T_canon[i:i+2, i:i+2]
            b = block[0, 1] # The top-right element is 'b'
            
            if b < 0:
                # If b is negative, apply a basis change (a swap) to flip its sign.
                # Update the T block: T_new =  T_old @ P
                T_canon[i:i+2, i:i+2] = P.T @ block @ P
                
                # Update the corresponding columns in U: U_new = U_old @ P
                U_canon[:, i:i+2] = U_canon[:, i:i+2] @ P
            
            i += 2 # Move past the 2x2 block
        else:
            # This is a 1x1 block
            i += 1
            
    return T_canon, U_canon

def schur_decomposition(matrix, return_original=False):
    """Compute a real Schur decomposition and return reordered basis data.

    Args:
        matrix: Real square matrix to decompose.
        return_original: If True, return the canonicalized Schur form and basis
            directly, without extracting and sorting scalar surrogate values.

    Returns:
        If ``return_original`` is True, returns ``(T_canon, Z_canon)``.
        Otherwise returns ``(sorted_scaling_values, sorted_basis)`` where the
        1x1 and 2x2 Schur blocks are converted into scalar surrogates and sorted
        in ascending order.
    """
    if not isinstance(matrix, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("Input must be a square matrix")
    
    # Convert to numpy for scipy compatibility
    matrix_np = matrix.cpu().numpy()

    # Perform Schur decomposition using scipy
    # T are the Schur form (upper quasi-triangular)
    # Z are the Schur vectors (orthogonal)
    T_np, Z_np = scipy.linalg.schur(matrix_np, output='real')
    T_np, Z_np = canonicalize_schur(T_np, Z_np)
    
    if return_original:
        return torch.from_numpy(T_np).float(), torch.from_numpy(Z_np).float()

    # Efficiently extract scalar diagonal values and indices using vectorized operations
    n = T_np.shape[0]
    
    # Pre-identify scalar vs 2x2 block elements
    is_scalar = np.zeros(n, dtype=bool)
    is_scalar[:-1] = np.isclose(T_np[1:, :-1].diagonal(), 0, atol=1e-5)
    is_scalar[-1] = True  # Last element is always scalar
    
    # Build arrays following the exact original logic
    scalar_diagonals = []
    scalar_index = []
    
    i = 0
    while i < n:
        if is_scalar[i]:
            # Scalar element: add once, increment by 1
            scalar_diagonals.append(np.abs(T_np[i, i]))
            scalar_index.append(i)
            i += 1
        else:
            # 2x2 block: add block_val twice, increment by 2
            block_val = np.sqrt(T_np[i, i]**2 + T_np[i+1, i]**2)
            scalar_diagonals.extend([block_val, block_val])
            scalar_index.extend([i, i+1])
            i += 2

    sorted_indices = np.argsort(scalar_diagonals)
    T_sorted = np.array(scalar_diagonals)[sorted_indices]
    Z_sorted = Z_np[:, sorted_indices]
    return torch.from_numpy(T_sorted).float(), torch.from_numpy(Z_sorted).float()

def svd_decomposition(matrix):
    """Compute a thin SVD and return singular vectors in descending order.

    Args:
        matrix: Input matrix of shape ``(m, n)``.

    Returns:
        A tuple ``(U, S, V)`` where ``U`` has shape ``(m, r)``, ``S`` has shape
        ``(r,)``, and ``V`` has shape ``(r, n)`` with ``r = min(m, n)``.
        The singular values and corresponding vectors are flipped so that the
        returned singular values are ordered from largest to smallest.
    """
    with torch.no_grad():
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        if matrix.dim() != 2:
            raise ValueError("Input must be a 2D matrix")
        
        # Ensure matrix is on the same device (CPU or CUDA)
        device = matrix.device
        dtype = matrix.dtype
        
        # Perform SVD decomposition using PyTorch
        svd_device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            U, S, V = torch.linalg.svd(matrix.to(svd_device), full_matrices=False)
        except RuntimeError as e:
            print(f"Using torch.svd instead of torch.linalg.svd")
            try:
                U, S, V = torch.svd(matrix.to(svd_device))
            except RuntimeError:
                raise RuntimeError(f"SVD decomposition failed: {e}")

        # Ensure all tensors are on the same device and dtype
        U = U.to(device=device, dtype=dtype)
        S = S.to(device=device, dtype=dtype)
        V = V.to(device=device, dtype=dtype).transpose(0, 1)
        
        
        
        # Reverse the order to get descending singular values
        U = torch.flip(U, dims=[1])  # Reverse columns of U
        S = torch.flip(S, dims=[0])  # Reverse singular values
        V = torch.flip(V, dims=[1])  # Reverse columns of V
    
    return U, S, V
    