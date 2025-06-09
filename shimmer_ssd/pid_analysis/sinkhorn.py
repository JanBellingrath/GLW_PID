import torch
import torch.utils.checkpoint
import gc
from typing import Optional

# Import coupling visualization for wandb logging
try:
    from .coupling_visualization import log_sinkhorn_coupling
    HAS_COUPLING_VIZ = True
except ImportError:
    HAS_COUPLING_VIZ = False



#TODO think about entropic regularization... I think Sinkhorn Knopp is without it, but check paper
# Global configurations (avoid circular imports by defining locally)
USE_AMP = False  # Default value
PRECISION = torch.float16  # Default precision for AMP
CHUNK_SIZE = 128  # Default chunk size
MEMORY_CLEANUP_INTERVAL = 10  # Clean memory every N chunks
AGGRESSIVE_CLEANUP = False  # Whether to aggressively clean memory

def sinkhorn_probs(
    matrix: torch.Tensor,
    x1_probs: torch.Tensor,
    x2_probs: torch.Tensor,
    tol: float = 1e-2, #TODO this is as in the NeurIPS paper and helps with vanishin gradients 
    max_iter: int = 5,  # Reduced from 10 to 5 for faster training
    chunk_size: Optional[int] = None, # Parameter chunk_size takes precedence if provided
    log_to_wandb: bool = False,  # New parameter for wandb logging
    wandb_prefix: str = "sinkhorn",  # Prefix for wandb logs
    wandb_log_interval: int = 1,  # Log every N iterations
    lr_finding_mode: bool = False,  # Skip expensive visualizations during LR finding
) -> torch.Tensor:
    """
    Apply the Sinkhorn-Knopp algorithm to project a matrix onto a transport polytope.
    Uses out-of-place operations to preserve gradients and chunked checkpointing.
    
    Args:
        matrix: Input matrix to project, shape [m, n]
        x1_probs: Target row sum distribution, shape [m]
        x2_probs: Target column sum distribution, shape [n]
        tol: Convergence tolerance for row/column sums
        max_iter: Maximum number of iterations
        chunk_size: Size of chunks for checkpointing (default: sqrt(max_iter) or CHUNK_SIZE if param is None)
        log_to_wandb: Whether to log coupling matrix visualizations to wandb during iterations
        wandb_prefix: Prefix for wandb log keys (default: "sinkhorn")
        wandb_log_interval: Log coupling matrix every N iterations (default: 50)
        lr_finding_mode: If True, skips expensive visualizations during LR finding (default: False)
        
    Returns:
        Projected matrix that approximately satisfies the marginal constraints
    """
    # Use local AMP and PRECISION settings
    if USE_AMP:
        dtype = PRECISION if matrix.is_cuda else torch.float32
    else:
        dtype = torch.float32
    
    matrix = matrix.to(dtype)
    x1_probs = x1_probs.to(dtype)
    x2_probs = x2_probs.to(dtype)
    
    if x2_probs.size(0) != matrix.size(1):
        raise ValueError(f"Dimension mismatch in Sinkhorn projection: matrix columns ({matrix.size(1)}) != x2_probs size ({x2_probs.size(0)}). This is likely an error.")

    if x1_probs.size(0) != matrix.size(0):
        raise ValueError(f"Dimension mismatch in Sinkhorn projection: matrix rows ({matrix.size(0)}) != x1_probs size ({x1_probs.size(0)}). This is likely an error.")

    #TODO check if this is truly sinkhorn knopp
    def sinkhorn_chunk(mat: torch.Tensor, n_steps: int) -> torch.Tensor:
        for _ in range(n_steps):
            col_sums = mat.sum(dim=0, keepdim=True, dtype=dtype)
            mat = mat / (col_sums + 1e-8) * x2_probs.unsqueeze(0) #TODO understand broadcasting rules
            row_sums = mat.sum(dim=1, keepdim=True, dtype=dtype)
            mat = mat / (row_sums + 1e-8) * x1_probs.unsqueeze(1)
        return mat
    #TODO urgenlty need to find out if there is a problem with oscillations in siinkhorn knopp, when there are on global constraints

    #TODO understand gradient checkpointing again
    # Determine effective chunk_size: parameter > original_default > global_fallback
    if chunk_size is not None:
        effective_chunk_size = chunk_size
    else: # chunk_size parameter is None, use original default behavior
        effective_chunk_size = int(max_iter ** 0.5)
        # As a secondary fallback, if one wanted to use CHUNK_SIZE when param is None and sqrt(max_iter) is not desired:
        # if CHUNK_SIZE is not None:
        #     effective_chunk_size = CHUNK_SIZE

    done = 0
    while done < max_iter:
        steps = min(effective_chunk_size, max_iter - done)
        matrix = torch.utils.checkpoint.checkpoint(
            lambda m, s=steps: sinkhorn_chunk(m, s),
            matrix,
            use_reentrant=False
        )
        done += steps
        
        # Log coupling matrix to wandb if requested (skip during LR finding)
        if log_to_wandb and HAS_COUPLING_VIZ and done % wandb_log_interval == 0 and not lr_finding_mode:
            log_sinkhorn_coupling(
                matrix, 
                step=done, 
                prefix=wandb_prefix,
                title_suffix=f"(iter {done}/{max_iter})",
                lr_finding_mode=lr_finding_mode
            )
        
        if (torch.allclose(matrix.sum(dim=1, dtype=dtype), x1_probs, atol=tol) and
            torch.allclose(matrix.sum(dim=0, dtype=dtype), x2_probs, atol=tol)):
            break
        
        # Use local MEMORY_CLEANUP_INTERVAL and AGGRESSIVE_CLEANUP
        if done % (effective_chunk_size * MEMORY_CLEANUP_INTERVAL) == 0 and AGGRESSIVE_CLEANUP:
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    # Log final coupling matrix to wandb if requested (skip during LR finding)
    if log_to_wandb and HAS_COUPLING_VIZ and not lr_finding_mode:
        log_sinkhorn_coupling(
            matrix, 
            step=done, 
            prefix=f"{wandb_prefix}_final",
            title_suffix=f"(converged at iter {done})",
            lr_finding_mode=lr_finding_mode
        )
    
    return matrix.to(torch.float32) 