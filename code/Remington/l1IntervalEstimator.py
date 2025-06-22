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
        grid_indices_here = grid_indices_here.to(device) * 1 / GRID  # Shape: (n_batch, n_inputs) Convert to radians
        n_inputs, n_batch = posterior.size() # n_inputs = len(\theta) (stimulus space), n_batch = len(F-1(m)) (encoding space)
        posterior = posterior.T.to(device)
        delta = 1 / GRID  # Distance between grid points

        # Initialize result tensor to store circular median for each m
        result = torch.zeros(n_batch)

        # Compute sin values for all combinations of (x_j, x_a) for all batches
        grid_radian_index = torch.arange(GRID, device=device) * 1 / GRID
        sin_values = (grid_radian_index.unsqueeze(1) - grid_radian_index.unsqueeze(0))   # shape: (n_inputs, n_inputs) sin[i][j] = sin(i - j)
        sin_values_expanded = sin_values.unsqueeze(0).expand(n_batch, -1, -1) # shape: (n_batch, n_inputs, n_inputs)

        # Compute A and B for all batches
        A = posterior  # Shape: (n_batch, n_inputs)
#        B = posterior.roll(GRID // 2, dims=0)  # Shape: (n_batch, n_inputs) Move the first (GRID // 2) elements to the end

        posterior_expanded = posterior.unsqueeze(-1).to(device)

        # Compute C and D for all (x_a) in parallel for all batches
        C = torch.where(sin_values_expanded > 0, posterior_expanded, torch.zeros_like(posterior_expanded)).sum(
            dim=1)  # Shape: (n_batch, n_inputs)
        D = torch.where(sin_values_expanded < 0, posterior_expanded, torch.zeros_like(posterior_expanded)).sum(
            dim=1)  # Shape: (n_batch, n_inputs)

        # Compute alpha for all batches and corresponding x_as
        denominator =  - 2 * A  # Shape: (n_batch, n_inputs)
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
        loss_matrix = torch.abs((theta_expanded - radian_indices_expanded)) * posterior.unsqueeze(
            1) * delta  # Shape: (n_batch, n_inputs, n_inputs)
        loss = torch.full_like(alpha, float('inf'), device=device)  # Shape: (n_batch, n_inputs) Initialize loss tensor to inf
        loss[valid_alpha_mask] = loss_matrix.sum(dim=2)[valid_alpha_mask]  # Compute loss only for valid alpha

        # Find the best theta with the smallest loss for each batch
        min_loss_idx = torch.argmin(loss, dim=1)  # Shape: (n_batch,)

        #print(loss)
        #numberOfSolutions = ((loss == torch.min(loss, dim=1, keepdim=True).values).float().sum(dim=1))
        #for i in range(0,GRID, 10):
        ##   if numberOfSolutions[i] > 1:
        #     print(posterior[:,i])
        #     # now plot xValues vs observations_y with ggplot
        #     import matplotlib.pyplot as plt
        #     import numpy as np
        #     # Create scatter plot
        #     plt.figure(figsize=(8, 6))
        #     #plt.hist(observations_y)
        #     plt.scatter(grid_radian_index, posterior[:,i], color='b', alpha=0.1, edgecolors='k', label='Data points')
        #     for j in range(GRID):
        #       if loss[i,j] == torch.min(loss, dim=1, keepdim=True).values[i]:
        #         plt.scatter([grid_radian_index[j]], [posterior[j,i]], color="red")
        #     # Labels and title
        #     plt.xlabel('X-axis')
        #     plt.ylabel('Y-axis')
        #     plt.title('Scatter Plot of Y vs X')
        #     plt.legend()
        #     plt.grid(True)
        #     
        #     # Show plot
        #     plt.show()
        #     plt.close()


        #quit()

        result = theta[torch.arange(n_batch, device=device), min_loss_idx]  # Shape: (n_batch,)

        ctx.save_for_backward(grid_indices_here, A, None, C, D, min_loss_idx)

        return result * GRID / (1)  # Convert back to grid scale

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
        #x_b = (x_a + (GRID // 2)) % GRID  # Shape: (n_batch, )

        # Get A_i, B_i, C_i, D_i Shape: (n_batch,)
        A_i = A[torch.arange(n_batch), x_a]
        #B_i = B[torch.arange(n_batch), x_a]
        C_i = C[torch.arange(n_batch), x_a]
        D_i = D[torch.arange(n_batch), x_a]

        # Broadcast to Shape: (n_batch, n_inputs)
        A_i_expand = A_i.unsqueeze(1).expand(n_batch, n_inputs)
        #B_i_expand = B_i.unsqueeze(1).expand(n_batch, n_inputs)
        C_i_expand = C_i.unsqueeze(1).expand(n_batch, n_inputs)
        D_i_expand = D_i.unsqueeze(1).expand(n_batch, n_inputs)

        # Construct mask:
        #    (1) x_j == x_a
        #    (2) x_j == x_b
        #    (3) sin(x_j - x_a) < 0
        #    (4) sin(x_j - x_a) > 0
        x_j = torch.arange(n_inputs, device=device).unsqueeze(0).expand(n_batch, n_inputs)  # Shape: (n_batch, n_inputs)
        mask_x_a = (x_j == x_a.unsqueeze(1))  # Shape: (n_batch, n_inputs)
#        mask_x_b = (x_j == x_b.unsqueeze(1))  # Shape: (n_batch, n_inputs)

        # sin_val[x, y] = sin(grid_indices_here[x_j, x_a])
        col_x_a = x_a.unsqueeze(1).expand(n_batch, n_inputs)  # Shape: (n_batch, n_inputs)
        sin_val = ((x_j - col_x_a) * 1 / GRID).to(device) # Shape: (n_batch, n_inputs)

        mask_sin_neg = (sin_val < 0)
        mask_sin_pos = (sin_val > 0)

        # Assign the gradient in order
        numerator = (D_i_expand - C_i_expand)  # Shape: (n_batch, n_inputs)
        denominator = 2.0 * ( - A_i_expand).pow(2)  # Shape: (n_batch, n_inputs)

        # Avoid ZeroDivision Error
        denominator = torch.where(denominator == 0, torch.full_like(denominator, float('inf')), denominator)

        gradient_x_a_val = numerator / denominator  # Shape: (n_batch, n_inputs)
#        gradient_x_b_val = - numerator / denominator  # Shape: (n_batch, n_inputs)

        # (1) x_j == x_a
        gradient[mask_x_a] = gradient_x_a_val[mask_x_a]

        # (2) x_j == x_b
#        gradient[mask_x_b] = gradient_x_b_val[mask_x_b]

        # (3) sin(x_j - x_a) < 0
        difference_BA = ( - A_i_expand)  # Shape: (n_batch, n_inputs)

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
