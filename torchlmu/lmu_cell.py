from typing import Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lmu_utils import lecun_uniform_, pytorch_cont2discrete_zoh, gen_AB_base_matrices

class LMUCell(nn.Module):
  """
    Implementation of LMU cell.

    The LMU cell consists of two parts: a memory component (decomposing
    the input signal using Legendre polynomials as a basis), and a hidden component
    (learning nonlinear mappings from the memory component). [1]_ [2]_

    This class processes one step within the whole time sequence input. Use the ``LMU``
    class to create a recurrent layer to process the whole sequence.

    References
    ----------
    .. [1] Voelker and Eliasmith (2018). Improving spiking dynamical
        networks: Accurate delays, higher-order synapses, and time cells.
        Neural Computation, 30(3): 569-609.
    .. [2] Voelker and Eliasmith. "Methods and systems for implementing
        dynamic neural networks." U.S. Patent Application No. 15/243,223.
        Filing date: 2016-08-22.
  """

  def __init__(
    self, 
    input_size:int, memory_d:int, order:int, hidden_size:int, theta:int,
    nonlinearity:partial=partial(torch.tanh),
    init_ex:partial=partial(lecun_uniform_), 
    init_eh:partial=partial(lecun_uniform_),
    init_em:partial=partial(nn.init.zeros_),
    init_Wx:partial=partial(nn.init.xavier_normal_), 
    init_Wh:partial=partial(nn.init.xavier_normal_),
    init_Wm:partial=partial(nn.init.xavier_normal_),
    learn_A=False,
    learn_B=False,
    hidden_to_memory=True,
    memory_to_memory=True,
    input_to_hidden=True,
  ):

    super().__init__()
    self.memory_d = memory_d
    self.order = order
    self._init_theta = theta
    self.hidden_size = hidden_size
    self.nonlinearity = nonlinearity
    self.hidden_to_memory = hidden_to_memory
    self.memory_to_memory = memory_to_memory
    self.input_to_hidden  = input_to_hidden

    # Discretized state space matrices A and B
    A, B = self._gen_AB()
    if learn_A:
        self.A = nn.Parameter(A)
    else:
        self.register_buffer("A", A)
    if learn_B:
        self.B = nn.Parameter(B)
    else:
        self.register_buffer("B", B)

    # Learnable encoding vectors: Convert/project input and state features into the 
    # signal (e.g., u(t)) that writes to the memory (see equation (7) in the paper).
    self.e_x = nn.Parameter(torch.empty(memory_d, input_size))
    init_ex(self.e_x)
    if self.hidden_to_memory:
        self.e_h = nn.Parameter(torch.empty(memory_d, hidden_size))
        init_eh(self.e_h)
    if self.memory_to_memory:
        self.e_m = nn.Parameter(torch.empty(memory_d, order * memory_d))
        init_em(self.e_m)

    # Learnable Kernels: Learn to compute non-linear functions across the memory
    # (see equation (6) in the original paper).
    if self.input_to_hidden:
        self.W_x = nn.Parameter(torch.empty(hidden_size, input_size))
        init_Wx(self.W_x)

    self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
    init_Wh(self.W_h)
    self.W_m = nn.Parameter(torch.empty(hidden_size, order * memory_d))
    init_Wm(self.W_m)


  def _gen_AB(self, dt=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    A, B = gen_AB_base_matrices(self.order)
    #self.register_buffer("_base_A", A)
    #self.register_buffer("_base_B", B)

    # Discretize
    Ad, Bd = pytorch_cont2discrete_zoh(A / self._init_theta, B / self._init_theta, dt)
    return Ad.float(), Bd.float()


  def forward(
    self, x:torch.Tensor, state:Tuple[torch.Tensor, torch.Tensor]
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # x: [batch_size, input_size]
    # state: (h, m)
    #   h: [batch_size, hidden_size]
    #   m: [batch_size, memory_d*order]
    h, m = state
    
    # equation (7) in the original paper
    u = F.linear(x, self.e_x)
    if self.hidden_to_memory:
        u = u + F.linear(h, self.e_h)
    if self.memory_to_memory:
        u = u + F.linear(m, self.e_m) 

    m = m.reshape(-1, self.memory_d, self.order)
    u = u.unsqueeze(-1)
    m = F.linear(m, self.A) + F.linear(u, self.B) # equation (4) in the original paper
    m = m.reshape(-1, self.memory_d * self.order)

    # equation (6) in the original paper
    if self.input_to_hidden:
        h = self.nonlinearity(
          F.linear(h, self.W_h) + 
          F.linear(m, self.W_m) + 
          F.linear(x, self.W_x)
        )
    else:
        h = self.nonlinearity(
          F.linear(h, self.W_h) + 
          F.linear(m, self.W_m)
        )
    
    return h, m
