from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

def lecun_uniform_(tensor: torch.Tensor) -> torch.Tensor:
    """ 
        LeCun Uniform Initializer
        References: 
        [1] https://keras.rstudio.com/reference/initializer_lecun_uniform.html
        [2] Source code of _calculate_correct_fan can be found in 
            https://pytorch.org/docs/stable/_modules/torch/nn/init.html
        [3] Yann A LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. 
            Efficient backprop. In Neural networks: Tricks of the trade, pages 9–48. 
            Springer, 2012
    """
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
    limit = np.sqrt(3.0 / fan_in)
    # Fills the tensor with values sampled from U(-limit, limit)
    return nn.init.uniform_(tensor, -limit, limit) 

def pytorch_cont2discrete_zoh(
    A: torch.Tensor, B: torch.Tensor, dt:float=1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Pytorch-specific implementation of discretization of a continuous-time 
        state space model using zero-order hold (ZOH) on the inputs.
    """
    em_upper = torch.cat((A, B), dim=1)
    # Need to stack zeros under the a and b matrices
    em_lower = torch.cat((
        torch.zeros((B.shape[1], B.shape[0]), dtype=A.dtype, device=A.device),
        torch.zeros((B.shape[1], B.shape[1]), dtype=A.dtype, device=A.device)
    ), dim=1)

    em = torch.cat((em_upper, em_lower), dim=0)
    ms = torch.linalg.matrix_exp(dt * em)

    # Dispose of the lower rows
    ms = ms[:A.shape[0], :]

    ad = ms[:, :A.shape[1]]
    bd = ms[:, A.shape[1]:]

    return ad, bd

def gen_AB_base_matrices(order:int) -> Tuple[torch.Tensor, torch.Tensor]:
   # Compute analog A/B matrices
   Q = torch.arange(order, dtype=torch.float64)
   R = (2 * Q + 1).unsqueeze(1)
   i, j = torch.meshgrid(Q, Q, indexing="ij")
   A = torch.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
   B = (-1.0) ** Q.unsqueeze(1) * R
   return A, B