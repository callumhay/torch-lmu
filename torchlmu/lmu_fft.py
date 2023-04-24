from typing import Tuple

import torch
import torch.nn as nn
from torch import fft

from .lmu_utils import pytorch_cont2discrete_zoh, gen_AB_base_matrices

class LMUFFT(nn.Module):

    def __init__(
        self, 
        input_size:int, 
        memory_d:int, 
        order:int, 
        hidden_size:int, 
        seq_len:int, 
        theta:int
    ):

        super(LMUFFT, self).__init__()

        self.hidden_size = hidden_size
        self.memory_d = memory_d
        self.order = order
        self.seq_len = seq_len
        self._init_theta = theta
        memory_size = memory_d * order

        self.W_u = nn.Linear(input_size, memory_d)
        self.f_u = nn.ReLU()
        self.W_h = nn.Linear(memory_size + input_size, hidden_size)
        self.f_h = nn.ReLU()

        A, B = self._gen_AB()
        self.register_buffer("A", A) # [memory_size, memory_size]
        self.register_buffer("B", B) # [memory_size, 1]

        H, fft_H = self._impulse()
        self.register_buffer("H", H) # [memory_size, seq_len]
        self.register_buffer("fft_H", fft_H) # [memory_size, seq_len + 1]


    def _gen_AB(self, dt=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        A, B = gen_AB_base_matrices(self.order)
        #self.register_buffer("_base_A", A)
        #self.register_buffer("_base_B", B)

        # Discretize
        Ad, Bd = pytorch_cont2discrete_zoh(
            A / self._init_theta, B / self._init_theta, dt
        )
        return Ad.float(), Bd.float()


    def _impulse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns the matrices H and the 1D Fourier transform of H 
           (Equations 23, 26 of the paper).
        """
        H = []
        A_i = torch.eye(self.order)
        for _ in range(self.seq_len):
            H.append(A_i @ self.B)
            A_i = self.A @ A_i
        H = torch.cat(H, dim = -1) # [order, seq_len]
        fft_H = fft.rfft(H, n = 2*self.seq_len, dim = -1) # [order, seq_len + 1]
        return H, fft_H


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, seq_len, input_size]
        """
        seq_len = x.shape[1]

        # Equation 18 of the paper
        u = self.f_u(self.W_u(x)) # [batch_size, seq_len, memory_d]

        # Equation 26 of the paper
        fft_input = u.permute(0, 2, 1) # [batch_size, memory_d, seq_len]
        fft_u = fft.rfft(fft_input, n = 2*seq_len, dim = -1) # [batch_size, memory_d, seq_len+1]

        # Element-wise multiplication (uses broadcasting)
        # [batch_size, memory_d, 1, seq_len+1] * [order, seq_len+1] =
        # [batch_size, memory_d, order, seq_len+1]
        temp = fft_u.unsqueeze(-2) * self.fft_H

        m = fft.irfft(temp, n = 2*seq_len, dim = -1)[..., :seq_len] # [batch_size, memory_d, order, seq_len]
        m = m.reshape(-1, self.memory_d*self.order, seq_len) # [batch_size, memory_d*order, seq_len]
        m = m.permute(0, 2, 1) # [batch_size, seq_len, memory_d*order]

        # Equation 20 of the paper (W_m@m + W_x@x  W@[m;x])
        input_h = torch.cat((m, x), dim = -1) # [batch_size, seq_len, memory_d*order + input_size]
        h = self.f_h(self.W_h(input_h)) # [batch_size, seq_len, hidden_size]

        h_n = h[:, -1, :] # [batch_size, hidden_size]

        return h, h_n