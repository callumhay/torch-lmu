from typing import Tuple
from functools import partial

import torch
import torch.nn as nn

from .lmu_utils import lecun_uniform_
from .lmu_cell import LMUCell

class LMU(nn.Module):
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
        learn_A:bool=False,
        learn_B:bool=False,
        hidden_to_memory:bool=True,
        memory_to_memory:bool=True,
        input_to_hidden:bool=True,
    ):
        super(LMU, self).__init__()
        self.hidden_size = hidden_size
        self.memory_d = memory_d
        self.order = order
        self.lmu_cell = LMUCell(
            input_size, memory_d, order, hidden_size, theta, nonlinearity, 
            init_ex, init_eh, init_em, init_Wx, init_Wh, init_Wm,
            learn_A=learn_A, learn_B=learn_B,
            hidden_to_memory=hidden_to_memory,
            memory_to_memory=memory_to_memory,
            input_to_hidden=input_to_hidden,
        )

    def forward(
      self, x:torch.Tensor, state:Tuple[torch.Tensor, torch.Tensor]|None=None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)
        seq_len    = x.size(1)

        if state is None:
            h_0 = torch.zeros(batch_size, self.hidden_size).to(x.device)
            m_0 = torch.zeros(batch_size, self.order * self.memory_d).to(x.device)
            state = (h_0, m_0)

        # Iterate over the timesteps
        output = []
        for t in range(seq_len):
            x_t = x[:, t, :] # [batch_size, input_size]
            h_t, m_t = self.lmu_cell(x_t, state)
            state = (h_t, m_t)
            output.append(h_t)
        
        output = torch.stack(output) # [seq_len, batch_size, hidden_size]
        output = output.permute(1, 0, 2)
        return output, state # state is the last output state of the sequence
    