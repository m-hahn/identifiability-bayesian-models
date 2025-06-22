import math
import random
import torch
import util
from util import MakeZeros
from util import savePlot
import numpy as np

class L1Estimator(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def set_parameters(**kwargs):
        global GRID
        GRID = kwargs["GRID"]
        global OPTIMIZER_VERBOSE
        OPTIMIZER_VERBOSE = kwargs["OPTIMIZER_VERBOSE"]

    @staticmethod
    def forward(ctx, grid_indices_here, posterior):
        """
        Forward pass: Compute the circular median for each fixed m using interpolation.
        """
        device = posterior.device
        
        # Convert to radians
        grid_indices_here = grid_indices_here.to(device) * 2 * math.pi / GRID  # Shape: (n_batch, n_inputs) Convert to radians
        n_inputs, n_batch = posterior.size() # n_inputs = len(\theta) (stimulus space), n_batch = len(F-1(m)) (encoding space)
        posterior = posterior.T.to(device)
        delta = 2 * math.pi / GRID  # Distance between grid points

        # Initialize result tensor to store circular median for each m
        result = torch.zeros(n_batch)

        # Compute sin values for all combinations of (x_j, x_a) for all batches
        grid_radian_index = torch.arange(GRID, device=device) * 2 * math.pi / GRID
        sin_values = torch.sin(grid_radian_index.unsqueeze(1) - grid_radian_index.unsqueeze(0))   # shape: (n_inputs, n_inputs) sin[i][j] = sin(i - j)
        sin_values_expanded = sin_values.unsqueeze(0).expand(n_batch, -1, -1) # shape: (n_batch, n_inputs, n_inputs)

        # Compute A and B for all batches
        A = posterior  # Shape: (n_batch, n_inputs)
        B = posterior.roll(GRID // 2, dims=0)  # Shape: (n_batch, n_inputs) Move the first (GRID // 2) elements to the end

        posterior_expanded = posterior.unsqueeze(-1).to(device)

        # Compute C and D for all (x_a) in parallel for all batches
        C = torch.where(sin_values_expanded > 0, posterior_expanded, torch.zeros_like(posterior_expanded)).sum(
            dim=1)  # Shape: (n_batch, n_inputs)
        D = torch.where(sin_values_expanded < 0, posterior_expanded, torch.zeros_like(posterior_expanded)).sum(
            dim=1)  # Shape: (n_batch, n_inputs)

        # Compute alpha for all batches and corresponding x_as
        denominator = 2 * B - 2 * A  # Shape: (n_batch, n_inputs)
        valid_mask = denominator != 0  # Shape: (n_batch, n_inputs) Avoid divide-by-zero
        alpha = torch.zeros_like(denominator, device=device) # Shape: (n_batch, n_inputs)
        alpha[valid_mask] = (D[valid_mask] - C[valid_mask]) / denominator[valid_mask]

        # Compute theta for valid alpha
        valid_alpha_mask = (alpha >= -0.5) & (alpha <= 0.5)  # Shape: (n_batch, n_inputs)
        # Initialize theta
        radian_indices = torch.arange(n_inputs, dtype=torch.float32, device=device) * delta
        radian_indices = radian_indices.unsqueeze(0).repeat(n_batch, 1) # Shape: (n_batch, n_inputs)
        theta = radian_indices.clone()  # Shape: (n_batch, n_inputs)

        # radian theta
        theta[valid_alpha_mask] = theta[valid_alpha_mask] + alpha[valid_alpha_mask] * delta

        # Compute losses for all valid theta
        theta_expanded = theta.unsqueeze(-1)  # Shape: (n_batch, n_inputs, 1)
        # Expanded radian indices for subtraction
        radian_indices_expanded = radian_indices.unsqueeze(1)  # Shape: (n_batch, 1, n_inputs)

        # Integrate on grids
        loss_matrix = torch.arccos(torch.cos(theta_expanded - radian_indices_expanded)) * posterior.unsqueeze(
            1) * delta  # Shape: (n_batch, n_inputs, n_inputs)
        loss = torch.full_like(alpha, float('inf'), device=device)  # Shape: (n_batch, n_inputs) Initialize loss tensor to inf
        loss[valid_alpha_mask] = loss_matrix.sum(dim=2)[valid_alpha_mask]  # Compute loss only for valid alpha

        # Find the best theta with the smallest loss for each batch
        min_loss, min_loss_idx = torch.min(loss, dim=1)  # Shape: (n_batch,)

        # solve ties circumstance
        tie_mask = (loss == min_loss.unsqueeze(1))  # Shape: (n_batch, n_inputs)
        tie_count = tie_mask.sum(dim=1)                  # Shape: (n_batch,)

        # initialize the final index
        best_idx = min_loss_idx.clone()  # Shape: (n_batch,)

        # find batches where appears ties
        tie_samples = (tie_count > 1) # Shape: (n_batch,)
        if tie_samples.any():
            # indices: batch index where appears ties
            indices = torch.nonzero(tie_samples).squeeze(1) # Shape: (n_ties, )
    
            # extract corresponding theta and posertior for these batches
            theta_sub = theta[indices]  # Shape: (n_ties, n_inputs)              
            posterior_sub = posterior[indices] # Shape: (n_ties, n_inputs)
    
            # calculate secondary loss 
            # cos_score[i, j] = sum_{k} posterior_sub[i,k] * cos( grid_radian_index[k] - theta_sub[i,j] )
            # extend theta and grid_radian_index
            theta_exp = theta_sub.unsqueeze(2)         # Shape: (n_ties, n_inputs, 1)
            grid_exp = grid_radian_index.view(1, 1, -1)  # Shape: (1, 1, n_inputs)
            diff = grid_exp - theta_exp                  # Shape: (n_ties, n_inputs, n_inputs)
            cos_term = torch.cos(diff)                   # Shape: (n_ties, n_inputs, n_inputs)
    
            # caluclate circular cosine score
            # extend posterior_sub to (n_ties, 1, n_inputs)，multiply cos_term and sum on the last dimension
            cos_score = torch.sum(posterior_sub.unsqueeze(1) * cos_term, dim=2)  # Shape: (n_ties, n_inputs)
    
            # convert to cos loss
            cos_loss = 1 - cos_score  # Shape: (n_ties, n_inputs)
    
            # keep ties candidates' cos loss
            # tie_mask Shape: (n_batch, n_inputs)
            tie_mask_sub = tie_mask[indices]  # Shape: (n_ties, n_inputs) 
            cos_loss[~tie_mask_sub] = float('inf')
    
            # for each sanple, select the min secondary loss index
            best_idx_sub = torch.argmin(cos_loss, dim=1)  # Shape: (n_ties,)
    
            # update best indices for these batches
            best_idx[indices] = best_idx_sub

        result = theta[torch.arange(n_batch, device=device), best_idx]  # Shape: (n_batch,)

        ctx.save_for_backward(grid_indices_here, A, B, C, D, best_idx)

        return result * GRID / (2 * math.pi)  # Convert back to grid scale

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        # Recover context
        # A, B, C, D Shape: (n_batch, n_inputs)
        # min_loss_idx Shape: (n_batch, )
        grid_indices_here, A, B, C, D, min_loss_idx = ctx.saved_tensors
        
        device = A.device

        n_batch, n_inputs = A.size()

        # Initialize gradient
        gradient = torch.zeros_like(A, device=device)  # Shape: (n_batch, n_inputs)

        # Calculate x_a, x_b for each batch
        # x_a: best index
        # x_b: antipodal point
        x_a = min_loss_idx # Shape: (n_batch, )
        x_b = (x_a + (GRID // 2)) % GRID  # Shape: (n_batch, )

        # Get A_i, B_i, C_i, D_i Shape: (n_batch,)
        A_i = A[torch.arange(n_batch), x_a]
        B_i = B[torch.arange(n_batch), x_a]
        C_i = C[torch.arange(n_batch), x_a]
        D_i = D[torch.arange(n_batch), x_a]

        # Broadcast to Shape: (n_batch, n_inputs)
        A_i_expand = A_i.unsqueeze(1).expand(n_batch, n_inputs)
        B_i_expand = B_i.unsqueeze(1).expand(n_batch, n_inputs)
        C_i_expand = C_i.unsqueeze(1).expand(n_batch, n_inputs)
        D_i_expand = D_i.unsqueeze(1).expand(n_batch, n_inputs)

        # Construct mask:
        #    (1) x_j == x_a
        #    (2) x_j == x_b
        #    (3) sin(x_j - x_a) < 0
        #    (4) sin(x_j - x_a) > 0
        x_j = torch.arange(n_inputs, device=device).unsqueeze(0).expand(n_batch, n_inputs)  # Shape: (n_batch, n_inputs)
        mask_x_a = (x_j == x_a.unsqueeze(1))  # Shape: (n_batch, n_inputs)
        mask_x_b = (x_j == x_b.unsqueeze(1))  # Shape: (n_batch, n_inputs)

        # sin_val[x, y] = sin(grid_indices_here[x_j, x_a])
        col_x_a = x_a.unsqueeze(1).expand(n_batch, n_inputs)  # Shape: (n_batch, n_inputs)
        sin_val = torch.sin((x_j - col_x_a) * 2 * math.pi / GRID).to(device) # Shape: (n_batch, n_inputs)

        mask_sin_neg = (sin_val < 0)
        mask_sin_pos = (sin_val > 0)

        # Assign the gradient in order
        numerator = (D_i_expand - C_i_expand)  # Shape: (n_batch, n_inputs)
        denominator = 2.0 * (B_i_expand - A_i_expand).pow(2)  # Shape: (n_batch, n_inputs)

        # Avoid ZeroDivision Error
        denominator = torch.where(denominator == 0, torch.full_like(denominator, float('inf')), denominator)

        gradient_x_a_val = numerator / denominator  # Shape: (n_batch, n_inputs)
        gradient_x_b_val = - numerator / denominator  # Shape: (n_batch, n_inputs)

        # (1) x_j == x_a
        gradient[mask_x_a] = gradient_x_a_val[mask_x_a]

        # (2) x_j == x_b
        gradient[mask_x_b] = gradient_x_b_val[mask_x_b]

        # (3) sin(x_j - x_a) < 0
        difference_BA = (B_i_expand - A_i_expand)  # Shape: (n_batch, n_inputs)

        # Avoid ZeroDivision Error
        difference_BA = torch.where(difference_BA == 0, torch.full_like(difference_BA, float('inf')), difference_BA)
        
        gradient[mask_sin_neg] = 1 / (2 * difference_BA[mask_sin_neg])

        # (4) sin(x_j - x_a) > 0
        gradient[mask_sin_pos] = -1 / (2 * difference_BA[mask_sin_pos])

        # Check and return gradient
        if OPTIMIZER_VERBOSE:
           if torch.isnan(grad_output).any():
                assert False, grad_output   # Shape: (n_batch, )

        # Transpose gradient
        gradient = gradient.T # Shape: (n_inputs, n_batch) gradient(x | F-1(m))
        # Add a new dimension and broadcast grad_output Shape: (1, n_batch)
        gradient = grad_output.unsqueeze(0) * gradient # Shape: (n_inputs, n_batch)

        if OPTIMIZER_VERBOSE:
           if torch.isnan(gradient).any():
                assert False, gradient

        return None, gradient
