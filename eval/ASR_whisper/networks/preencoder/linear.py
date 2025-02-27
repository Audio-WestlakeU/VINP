#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear Projection."""

from typing import Tuple

import torch
# from typeguard import typechecked

from networks.preencoder.abs_preencoder import AbsPreEncoder


class LinearProjection(AbsPreEncoder):
    """Linear Projection Preencoder."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        """Initialize the module."""
        # assert typechecked()
        super().__init__()

        self.output_dim = output_size
        self.linear_out = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        output = self.linear_out(self.dropout(input))
        return output, input_lengths  # no state in this layer

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim
