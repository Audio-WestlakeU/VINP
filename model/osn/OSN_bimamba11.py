from typing import *
import torch
import torch.nn as nn
from torch import Tensor
from .base.linear_group import LinearGroup
from .base.non_linear import *
from .base.norm import *
from mamba_ssm import Mamba as Mamba


class SpatialNetLayer(nn.Module):

    def __init__(
        self,
        dim_hidden: int,
        dim_squeeze: int,
        num_freqs: int,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "LN", "LN", "LN", "LN"],
        padding: str = "zeros",
        full: nn.Module = None,
        attention: str = "mamba(16,4)",
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]

        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList(
            [
                new_norm(
                    norms[3],
                    dim_hidden,
                    seq_last=True,
                    group_size=None,
                    num_groups=f_conv_groups,
                ),
                nn.Conv1d(
                    in_channels=dim_hidden,
                    out_channels=dim_hidden,
                    kernel_size=f_kernel_size,
                    groups=f_conv_groups,
                    padding="same",
                    padding_mode=padding,
                ),
                nn.PReLU(dim_hidden),
            ]
        )
        # full-band linear module
        self.norm_full = new_norm(
            norms[5],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=f_conv_groups,
        )
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1),
            nn.SiLU(),
        )
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = (
            LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze)
            if full == None
            else full
        )
        self.unsqueeze = nn.Sequential(
            nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1),
            nn.SiLU(),
        )
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList(
            [
                new_norm(
                    norms[4],
                    dim_hidden,
                    seq_last=True,
                    group_size=None,
                    num_groups=f_conv_groups,
                ),
                nn.Conv1d(
                    in_channels=dim_hidden,
                    out_channels=dim_hidden,
                    kernel_size=f_kernel_size,
                    groups=f_conv_groups,
                    padding="same",
                    padding_mode=padding,
                ),
                nn.PReLU(dim_hidden),
            ]
        )

        # narrow-band block
        # MHSA module
        self.norm_mhsa = new_norm(
            norms[0],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=t_conv_groups,
        )

        attn_params = attention[6:-1].split(",")
        d_state, mamba_conv_kernel = int(attn_params[0]), int(attn_params[1])
        self.mhsa_f = Mamba(
            d_model=dim_hidden,
            d_state=d_state,
            d_conv=mamba_conv_kernel,
            expand=2,
        )
        # layer_idx=0)

        self.attention = attention
        self.dropout_mhsa = nn.Dropout(dropout[0])
        # T-ConvFFN module

        self.norm_tconvffn = new_norm(
            norms[1],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=t_conv_groups,
        )
        self.tconvffn_b = Mamba(
            d_model=dim_hidden,
            d_state=d_state,
            d_conv=mamba_conv_kernel,
            expand=2,
        )
        # layer_idx=0)

        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)
        x = x + self._mamba(x, self.mhsa_f, self.norm_mhsa, self.dropout_mhsa)

        x = x + self._mamba(
            x.flip(-2), self.tconvffn_b, self.norm_tconvffn, self.dropout_tconvffn
        ).flip(-2)

        return x

    def _mamba(self, x: Tensor, mamba, norm: nn.Module, dropout: nn.Module):
        B, F, T, H = x.shape
        x = norm(x)
        x = x.reshape(B * F, T, H)

        x = mamba.forward(x)
        x = x.reshape(B, F, T, H)
        return dropout(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3)  # [B,T,H',F]
            x = x.reshape(B * T, -1, F)

        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


class BiSpatialNet(nn.Module):

    def __init__(
        self,
        dim_input: int,  # the input dim for each time-frequency point
        dim_output: int,  # the output dim for each time-frequency point
        num_layers: int,
        dim_squeeze: int,
        num_freqs: int,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "GN", "LN", "LN", "LN"],
        padding: str = "zeros",
        full_share: int = 0,  # share from layer 0
        attention: str = "mhsa(251)",  # mhsa(frames), ret(factor)
    ):
        super().__init__()

        self.padding_size = (0, (encoder_kernel_size - 1) // 2)
        # encoder
        self.encoder = nn.Conv2d(
            in_channels=dim_input,
            out_channels=dim_hidden,
            padding=0,
            kernel_size=(1, encoder_kernel_size),
        )
        # self.encoder = NonCausalConv1d(in_channels=dim_input,
        #                             out_channels=dim_hidden,
        #                             kernel_size=encoder_kernel_size)

        # spatialnet layers
        full = None
        layers = []
        for l in range(num_layers):
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
                attention=attention,
            )
            if hasattr(layer, "full"):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)

    def forward(self, input: Tensor) -> Tensor:

        # mean = input.mean(dim=(1, 2, 3), keepdim=True)
        # std = input.std(dim=(1, 2, 3), keepdim=True)
        # input = (input - mean) / std

        input_pad = torch.nn.functional.pad(
            input,
            (
                self.padding_size[1],
                self.padding_size[1],
                self.padding_size[0],
                self.padding_size[0],
            ),
            mode="constant",
            value=-8,
        )

        x = self.encoder(input_pad)

        x = x.permute(0, 2, 3, 1)

        B, F, T, H = x.shape

        for _, m in enumerate(self.layers):
            x = m(x)

        y = self.decoder(x)

        y = y.permute(0, 3, 1, 2)

        return y.contiguous()


if __name__ == "__main__":
    model = BiSpatialNet(
        dim_input=1,
        dim_output=1,
        num_layers=8,
        dim_hidden=96,
        dim_ffn=192,
        num_heads=4,
        encoder_kernel_size=1,
        kernel_size=[5, 3],
        conv_groups=[8, 8],
        dropout=[0, 0, 0],
        norms=["LN", "LN", "LN", "LN", "LN", "LN"],
        dim_squeeze=8,
        num_freqs=257,
        full_share=0,
        attention="mamba(16,4)",
        rope=False,
    ).cuda()
    print(model)

    x = torch.randn((1, 1, 257, 500)).cuda()  # 6-channel, 4s, 8 kHz
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(model, display=False) as fcm:
        res = model(x, inference=True).mean()
        flops_forward_eval = fcm.get_total_flops()
    for k, v in fcm.get_flop_counts().items():
        ss = f"{k}: {{"
        for kk, vv in v.items():
            ss += f" {str(kk)}:{vv}"
        ss += " }"
        print(ss)
    params_eval = sum(param.numel() for param in model.parameters())
    print(
        f"flops_forward={flops_forward_eval/4e9:.2f}G/s, params={params_eval/1e6:.2f} M"
    )
