import torch.nn as nn
import torch
import math

mse_loss = nn.MSELoss


class JS_loss(nn.Module):

    def __init__(self, eps, *args, **kwargs) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        B, C, F, T = x.shape
        x = x**2 + self.eps
        y = y**2 + self.eps
        ret = 1 / 2 * (x / y + y / x) - 1

        return torch.sum(ret) / B / T / F


class KL_loss(nn.Module):

    def __init__(self, eps, bidir=False, *args, **kwargs) -> None:
        super().__init__()
        self.eps = eps
        self.bidir = bidir

    def forward(self, y, x):
        B, C, F, T = x.shape
        assert C == 1
        x = x + self.eps
        y = y + self.eps
        # x = (x**2).clamp(self.eps)
        # y = (y**2).clamp(self.eps)
        if self.bidir == False:
            ret = torch.sum(torch.log(y) - torch.log(x) + x / y - 1)
        elif self.bidir == True:
            ret = (
                torch.sum(torch.log(y) - torch.log(x) + x / y - 1)
                + torch.sum(torch.log(x) - torch.log(y) + y / x - 1)
            ) / 2

        return ret / B / T / F
    
class KL_mse_loss(nn.Module):

    def __init__(self, eps, bidir=False, *args, **kwargs) -> None:
        super().__init__()
        self.eps = eps
        self.bidir = bidir

    def forward(self, y, x):
        B, C, F, T = x.shape
        assert C == 1
        mse = (x.pow(1/3)-y.pow(1/3)).pow(2).mean()
        x = x.abs()**2 + self.eps
        y = y.abs()**2 + self.eps
        # x = (x**2).clamp(self.eps)
        # y = (y**2).clamp(self.eps)
        if self.bidir == False:
            ret = torch.sum(torch.log(y) - torch.log(x) + x / y - 1)
        elif self.bidir == True:
            ret = (
                torch.sum(torch.log(y) - torch.log(x) + x / y - 1)
                + torch.sum(torch.log(x) - torch.log(y) + y / x - 1)
            ) / 2

        return ret / B / T / F+mse*1

class KL_loss_meanscale(nn.Module):

    def __init__(self, eps, bidir=False, *args, **kwargs) -> None:
        super().__init__()
        self.eps = eps
        self.bidir = bidir

    def forward(self, y, x):
        B, C, F, T = x.shape
        assert C == 1
        x = x**2
        scale = x.mean()*self.eps
        y = y**2
        x = x + scale
        y = y+scale
        # x = (x**2).clamp(self.eps)
        # y = (y**2).clamp(self.eps)
        if self.bidir == False:
            ret = torch.sum(torch.log(y) - torch.log(x) + x / y - 1)
        elif self.bidir == True:
            ret = (
                torch.sum(torch.log(y) - torch.log(x) + x / y - 1)
                + torch.sum(torch.log(x) - torch.log(y) + y / x - 1)
            ) / 2

        return ret / B / T / F

class KL_loss_maxscale(nn.Module):

    def __init__(self, eps, *args, **kwargs) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, y, x):
        B, C, F, T = x.shape
        assert C == 1
        x = x**2
        eps,_ = x.max(3,keepdim=True)
        eps,_ = eps.max(2,keepdim=True)
        eps,_ = eps.max(1,keepdim=True)
        eps = eps*self.eps
        y = y**2
        x = x + eps
        y = y + eps
        ret = torch.sum(torch.log(y) - torch.log(x) + x / y - 1)


        return ret / B / T / F

class WD_loss(nn.Module):

    def __init__(self) -> None:
        super(WD_loss, self).__init__()

    def forward(self, x, y):
        # x = x.exp()
        # y = y.exp()

        ret = (torch.sqrt(x) - torch.sqrt(y)).pow(2)

        return torch.mean(ret)


class IS_loss(nn.Module):

    def __init__(self, eps=1e-4, *args, **kwargs) -> None:
        super(IS_loss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        B, C, F, T = target.shape
        assert C == 1
        target = target.abs()**2 + self.eps
        output = output.abs()**2 + self.eps
        # target = (target**2).clamp(self.eps)
        # output = (output**2).clamp(self.eps)

        ret = torch.sum(target / output - torch.log(target) + torch.log(output) - 1)

        return ret / B / T / F


class Mix_IS_loss(nn.Module):

    def __init__(self, eps=[1e-4], weight=[1], *args, **kwargs) -> None:
        super(Mix_IS_loss, self).__init__()
        self.eps = eps
        self.weight = weight
        assert len(eps) == len(weight)
        self.n_mix = len(eps)

    def forward(self, output, target):

        for n in range(self.n_mix):
            target_this = target**2 + self.eps[n]
            output_this = output**2 + self.eps[n]

            if n == 0:
                ret = (
                    torch.mean(
                        target_this / output_this
                        - torch.log(target_this)
                        + torch.log(output_this)
                        - 1
                    )
                    * self.weight[n]
                )
            else:
                ret += (
                    torch.mean(
                        target_this / output_this
                        - torch.log(target_this)
                        + torch.log(output_this)
                        - 1
                    )
                    * self.weight[n]
                )

        return ret


class log_MSE_loss(nn.Module):

    def __init__(self, eps=1e-6, *args, **kwargs):
        super(log_MSE_loss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        target = (target.abs() + self.eps).log()
        output = (output.abs() + self.eps).log()

        ret = torch.mean((output - target) ** 2)

        return ret


class KL_IS_loss(nn.Module):

    def __init__(
        self,
        zero_step: int = 1000,
        warmup_step: int = 500,
        hold_step: int = 500,
        beta: float = 1,
        eps: float = 1e-16,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.zero_step = zero_step
        self.warmup_step = warmup_step
        self.hold_step = hold_step
        self.beta = beta
        self.eps = eps
        # assert beta <= 1 and beta >= 0

    def IS_loss(self, output_abs: torch.Tensor, target_abs: torch.Tensor):
        """
        Return: IS loss
        Params:
            output: magnitude spectrogram (>=0), output of RVAE decoder, shape of [bs,T,F]
            target: magnitude spectrogram (>=0), clean, shape of [bs,T,F]
        """
        B, C, F, T = output_abs.shape
        assert C == 1
        target_abs = target_abs**2 + self.eps
        output_abs = output_abs**2 + self.eps
        ret = torch.sum(
            target_abs / output_abs - torch.log(target_abs) + torch.log(output_abs) - 1
        )

        return ret / B / T

    def KL_loss(self, zmean: torch.Tensor, zlogvar: torch.Tensor):
        """
        Return: KL loss
        Params:
            zmean: mean of latent variables, output of RVAE encoder, shape of [bs,T,D]
            zlogvar: log variance of latent variables, output of RVAE encoder, shape of [bs,T,D]
        """
        B, C, F, T = zmean.shape
        assert C == 1
        # zmean_p = torch.zeros_like(zmean)
        # zlogvar_p = torch.zeros_like(zlogvar)
        ret = -0.5 * torch.sum(
            zlogvar
            - zlogvar.exp()
            - zmean.pow(2)
            + 1
            # - zlogvar_p
            # - torch.div(zlogvar.exp() + (zmean - zmean_p).pow(2), zlogvar_p.exp())
            # + 1
        )
        return ret / B / T

    def cal_KL_scale(self, cur_step: int):
        """
        Return: the scale of KL loss
        Params:
            cur_step: current training step
            beta: base scale (default: 1)
            zero_step: keep scale = 0
            warmup_step: scale linear increased
            hold_step: keep scale=1
        """
        period = self.warmup_step + self.hold_step
        if cur_step < self.zero_step:
            return 0
        else:
            epoch_mod = (cur_step - self.zero_step) % period
            if epoch_mod < self.warmup_step:
                return self.beta * epoch_mod / self.warmup_step
                # return self.beta * math.sin(epoch_mod / self.warmup_step * math.pi / 2)
            else:
                return self.beta
                # return self.beta * math.cos(
                #     (epoch_mod - self.warmup_step) / self.hold_step * math.pi / 2
                # )

    def forward(
        self, output_abs, target_abs, zmean, zlogvar, curr_step, isval: bool = False
    ):
        # print(output_abs.shape,target_abs.shape,zmean.shape,zlogvar.shape)
        # exit()
        if isval:
            # ret = self.IS_loss(output_abs, target_abs) + self.KL_loss(zmean, zlogvar)
            ret = self.IS_loss(output_abs, target_abs) + self.KL_loss(zmean, zlogvar)
        else:
            KL_scale = self.cal_KL_scale(curr_step)
            ret = self.IS_loss(output_abs, target_abs) + KL_scale * self.KL_loss(
                zmean, zlogvar
            )

        return ret
