# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: VEM algorithm.

import torch
from torch import vmap
from torch.nn.functional import pad
import numpy as np
import torchaudio


class VEM:
    def __init__(
        self,
        CTF_len,
        errvar_init,
        max_steps,
        min_steps,
        eda_factor_mu,
        eda_factor_var,
        eda_factor_CTF,
        eda_factor_ErrVar,
        sr=16000,
        smooth_pre: int = 0,
        *args,
        **kwargs
    ):
        """
        Initialization
        """
        self.L = CTF_len
        self.errvar_init = errvar_init
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.eda_factor_mu = eda_factor_mu
        self.eda_factor_var = eda_factor_var
        self.eda_factor_CTF = eda_factor_CTF
        self.eda_factor_ErrVar = eda_factor_ErrVar
        self.patience = 5
        self.sr = sr
        self.smooth_pre = smooth_pre

        self.E_step_f_t_para = vmap(self.E_step_f_t)
        self.E_step_f_para = vmap(self.E_step_f)
        self.M_step_f_t_ErrVar_para = vmap(self.M_step_f_t_ErrVar)
        self.M_step_f_t_CTF_para = vmap(self.M_step_f_t_CTF)
        self.M_step_f_ErrVar_para = vmap(self.M_step_f_ErrVar)
        self.M_step_f_CTF_para = vmap(self.M_step_f_CTF)
        self.cal_likeli_f_t_para = vmap(self.cal_likeli_f_t)
        self.cal_likeli_f_para = vmap(self.cal_likeli_f)

        # log-sinesweep
        pi = np.pi
        duration = 5
        f1 = 100
        f2 = self.sr / 2
        w1 = 2 * pi * f1 / self.sr
        w2 = 2 * pi * f2 / self.sr
        num_sample = int(duration * self.sr)
        sinesweep = np.zeros(num_sample)
        taxis = np.arange(0, num_sample, 1) / (num_sample - 1)

        lw = np.log(w2 / w1)
        sinesweep = np.sin(w1 * (num_sample - 1) / lw * (np.exp(taxis * lw) - 1))

        envelope = (w2 / w1) ** (-taxis)

        invfilter = np.flipud(sinesweep) * envelope
        scaling = (
            pi
            * num_sample
            * (w1 / w2 - 1)
            / (2 * (w2 - w1) * np.log(w1 / w2))
            * (w2 - w1)
            / pi
        )

        self.sinesweep = torch.from_numpy(sinesweep).float()
        self.invfilter = torch.from_numpy(invfilter / scaling).float()

    def init_seg(self, rev_spec, clean_power, Noise_power, *args, **kwargs):
        """
        对每段音频的初始化
        input:
            rev_spec: [F,T] complex
            clean_spec: [F,T] complex or real
        return:
            Obs: [F,T] complex 观测频谱
            Sig_var: [F,T] real 干净语音先验方差
            Noi_var: [F,T] real 噪声先验方差
            CTF: [F,L] complex CTF滤波器
            Err_var: [F] real 误差方差
        """

        Obs = rev_spec  # [F,T] complex
        self.dtype = Obs.dtype
        self.device = Obs.device

        F, T = Obs.shape
        Sig_var = clean_power  # [F,T]

        Sig_var.to(self.device)

        if self.smooth_pre > 0:
            Sig_var_pad = Sig_var[:, 0].unsqueeze(1).repeat([1, self.smooth_pre])
            Sig_var = torch.concat([Sig_var_pad, Sig_var], -1)

            smooth_filter = (
                torch.ones([F, self.smooth_pre], device=self.device) / self.smooth_pre
            )
            Sig_var = torchaudio.functional.fftconvolve(Sig_var, smooth_filter)[
                :, self.smooth_pre : self.smooth_pre + T
            ]

        # Sig_var = torch.min(Sig_var,Obs.abs().pow(2))

        CTF = torch.zeros([F, self.L], device=self.device, dtype=self.dtype)
        CTF[:, -1] = CTF[:, -1] + 1

        MIN_OBS_VAR, _ = Obs.abs().pow(2).min(1)

        Err_var = MIN_OBS_VAR * self.errvar_init + 1e-16

        Mu = torch.zeros([F, T], device=self.device, dtype=self.dtype)

        Var = (Obs.abs() ** 2) + 1e-16

        return (Obs, Sig_var, CTF, Err_var, Mu, Var)

    @torch.no_grad()
    def process(self, Obs_wav, model, TF, device, *args, **kwargs):

        Obs_wav = (Obs_wav.unsqueeze(0).unsqueeze(0).to(device)).float()

        sinesweep = self.sinesweep.to(device)
        invfilter = self.invfilter.to(device)

        sinesweep_spec = TF.stft(sinesweep, "complex")

        scale = Obs_wav.abs().max() + 1e-32
        Obs_wav /= scale
        Obs = TF.stft(Obs_wav, "complex")
        Obs_abs = Obs.abs()

        Obs_ft = TF.preprocess(Obs_abs)

        Clean_ft = model(Obs_ft)

        Clean_abs = TF.postprocess(Clean_ft)
        Clean_power = Clean_abs**2
        Noise_power = torch.zeros_like(Clean_power)

        B, M, F, T = Obs.shape
        assert B == 1
        assert M == 1

        Obs_all = Obs.squeeze(0, 1)
        Clean_power_all = Clean_power.squeeze(0, 1)
        Noise_power = Noise_power.squeeze(0, 1)

        Clean_power_all = pad(
            Clean_power_all, (0, Obs_all.shape[-1] - Clean_power_all.shape[-1])
        )
        Noise_power = pad(Noise_power, (0, Obs_all.shape[-1] - Noise_power.shape[-1]))

        result = torch.zeros_like(Obs_all)

        time_start = 0
        time_stop = T

        Obs = Obs_all[:, time_start:time_stop]

        Clean_power = Clean_power_all[:, time_start:time_stop]
        Noise_power = Noise_power[:, time_start:time_stop]

        Obs, Sig_var, CTF, Err_var, Mu, Var = self.init_seg(
            Obs, Clean_power, Noise_power
        )

        loglikeli = -1e32
        patience = 0
        Sig_var_smooth = Sig_var.clone()

        for istep in range(self.max_steps):
            Mu_old = Mu
            Var_old = Var
            CTF_old = CTF
            Err_var_old = Err_var
            Mu, Var, _ = self.E_step_f_para(
                Obs, Sig_var_smooth, CTF, Err_var, Mu, Noise_power
            )

            Mu = (1 - self.eda_factor_mu) * Mu + self.eda_factor_mu * Mu_old
            Var = (
                (1 - self.eda_factor_var) * Var + self.eda_factor_var * Var_old + 1e-16
            )

            CTF = self.M_step_f_CTF_para(Obs, CTF, Mu, Var)
            CTF = (1 - self.eda_factor_CTF) * CTF + self.eda_factor_CTF * CTF_old

            Err_var = self.M_step_f_ErrVar_para(Obs, CTF, Mu, Var, Noise_power)
            Err_var = (
                (1 - self.eda_factor_ErrVar) * Err_var
                + self.eda_factor_ErrVar * Err_var_old
                + 1e-16
            )

            loglikeli_now = self.cal_likeli_f_para(Obs, CTF, Mu, Var, Err_var).mean()
            if loglikeli_now >= loglikeli:
                loglikeli = loglikeli_now
                Mu_ret = Mu
                CTF_ret = CTF
                patience = 0

            elif loglikeli_now < loglikeli and patience < self.patience:
                patience += 1
            elif loglikeli_now < loglikeli and patience >= self.patience:
                break

        result[:, time_start:time_stop] = Mu_ret
        result[0:3, :] *= 0

        CTF_ret[0:3, :] *= 0
        CTF_ret = CTF_ret.unsqueeze(2)  # [F,1,T]

        sinesweep_spec = pad(sinesweep_spec, (self.L - 1, 0))  # [F,T+L-1]
        sinesweep_spec = sinesweep_spec.unfold(1, self.L, 1)  # [F,T,L] complex

        ir_spec = torch.matmul(sinesweep_spec, CTF_ret).squeeze()
        ir = TF.istft(ir_spec, "complex")

        rir = torchaudio.functional.fftconvolve(invfilter, ir, mode="full").to("cpu")

        rir = rir[torch.argmax(rir.abs()) - int(0.0025 * self.sr) :]
        if rir.abs().max() > 1:
            rir /= rir.abs().max()

        result_wav = TF.istft(result, "complex")
        result_wav = result_wav.squeeze()
        result_wav *= scale

        result_wav = result_wav.to("cpu")

        return result_wav, rir, istep + 1, loglikeli

    @torch.no_grad()
    def E_step_f(
        self, Obs_f, Sig_var_f, CTF_f, Err_var_f, mu_f, NoiVar_f, *args, **kwargs
    ):
        """
        对每个频点进行E步后验计算
        input:
            Obs_f: [T] complex
            Sig_var_f: [T] real
            CTF_f: [L] complex
            Err_var_f: [] real
            mu_f: [T] complex
            var_f: [T] real
        return:
            mu_f_ret: [T] complex
            var_f_ret: [T] real
        """
        T = Obs_f.shape[0]
        Obs_f = pad(Obs_f, (0, self.L - 1))
        Obs_f_para = Obs_f.unfold(0, self.L, 1)  # [T,L] complex
        Sig_var_f_para = Sig_var_f  # [T] real
        CTF_f_para = CTF_f.unsqueeze(0).repeat(T, 1)  # [T,L] complex
        Err_var_f_para = Err_var_f.unsqueeze(0).repeat(T) + NoiVar_f  # [T] real
        mu_f = pad(mu_f, (self.L - 1, self.L - 1))  # [T+2L-2]
        mu_f_para = mu_f.unfold(0, self.L * 2 - 1, 1)  # [T,2L-1] complex

        mu_f_ret, var_f_ret, err_f_est = self.E_step_f_t_para(
            Obs_f_para,
            Sig_var_f_para,
            CTF_f_para,
            Err_var_f_para,
            mu_f_para,
        )

        return mu_f_ret, var_f_ret.real, err_f_est

    @torch.no_grad()
    def E_step_f_t(
        self, Obs_f_t, Sig_var_f_t, CTF_f, Err_var_f_t, mu_f_t, *args, **kwargs
    ):
        """
        对每个时频点进行E步后验计算
        input:
            Obs_f_t [L] complex t:t+L-1
            Sig_var_f_t [] real
            CTF_f [L] complex from last iteration
            Err_var_f_t [] real from last iteration
            mu_f_t [2L-1] complex t-L+1:t+L-1 from last iteration
            var_f_t [] real from last iteration
            MACs: 4L^2+6L+6
        return:
            mu_f_t [] complex
            var_f_t [] real
        """
        var_f_t = (
            Sig_var_f_t.pow(-1) + Err_var_f_t.pow(-1) * CTF_f.abs().pow(2).sum()
        ).real.pow(
            -1
        )  # m:L

        Mu_f_t = []
        for l in range(self.L):
            mu_f_t_added = mu_f_t[l : l + self.L].clone()
            mu_f_t_added[-l - 1] = torch.tensor(0, device=self.device, dtype=self.dtype)

            Mu_f_t.append(mu_f_t_added)
        Mu_f_t = torch.stack(Mu_f_t, dim=1)
        Err_f_t = Obs_f_t - torch.matmul(CTF_f.unsqueeze(0), Mu_f_t).squeeze()

        mu_f_t = (
            var_f_t * Err_var_f_t.pow(-1) * ((CTF_f.conj() * Err_f_t.flip(0)).sum())
        )

        return mu_f_t, var_f_t, Err_f_t[0]

    @torch.no_grad()
    def M_step_f_t_ErrVar(self, Obs_f_t, CTF_f, mu_f_t, var_f_t, *args, **kwargs):
        """
        对每个时频点估计CTF和噪声方差
        input:
            Obs_f_t: [] complex
            CTF_f: [L] complex
            mu_f_t: [L] complex
            var_f_t: [L] real
        return:
            Err_var_f_t: [] real
            CTF_f_t_PartA: [L,L] complex
            CTF_f_t_PartB: [L,1] complex

        Number of multiplications: 12L+1
        Number of addition: 7L
        """

        Err_var_f_t = Obs_f_t.abs().pow(2)  # m:1 a:0
        temp = (CTF_f * mu_f_t).sum()  # m:4L a:L
        Err_var_f_t = (
            Err_var_f_t
            + temp.abs().pow(2)  # m:L a:0
            + (CTF_f.abs().pow(2) * var_f_t).sum()  # m:2L a:L
        )  # m:0 a:2L
        Err_var_f_t = Err_var_f_t - 2 * (Obs_f_t.conj() * temp).real  # m:5L a:3L

        return Err_var_f_t.real

    @torch.no_grad()
    def M_step_f_t_CTF(self, Obs_f_t, CTF_f, mu_f_t, var_f_t, *args, **kwargs):
        """
        对每个时频点估计CTF和噪声方差
        input:
            Obs_f_t: [] complex
            CTF_f: [L] complex
            mu_f_t: [L] complex
            var_f_t: [L] real
        return:
            Err_var_f_t: [] real
            CTF_f_t_PartA: [L,L] complex
            CTF_f_t_PartB: [L,1] complex
        Number of multiplications: 4L^2+4L
        Number of addition: 2L^2+3L
        """

        CTF_f_t_PartA = torch.matmul(
            mu_f_t.unsqueeze(1), mu_f_t.unsqueeze(0).conj()
        ) + torch.diag_embed(
            var_f_t
        )  # m:4L^2 a:2L^2+L

        CTF_f_t_PartB = Obs_f_t * mu_f_t.conj().unsqueeze(0)  # m:4L a:2L

        return CTF_f_t_PartA, CTF_f_t_PartB

    @torch.no_grad()
    def M_step_f_ErrVar(self, Obs_f, CTF_f, mu_f, var_f, NoiVar_f, *args, **kwargs):
        """
        对每个频带估计CTF和噪声方差
        input:
            Obs_f: [T] complex
            CTF_f: [L] complex
            mu_f: [T] complex
            var_f: [T] real
        return:
            CTF_f_ret: [L] complex
            Err_var_f_ret: [] real
        Number of multiplications: T(12L+1)
        Number of addition: T(7L+2)
        """
        T = Obs_f.shape[0]
        Obs_f_para = Obs_f  # [T] complex
        CTF_f_para = CTF_f.unsqueeze(0).repeat(T, 1)  # [T,L] complex
        mu_f = pad(mu_f, (self.L - 1, 0))  # [T+L-1]
        mu_f_para = mu_f.unfold(0, self.L, 1)  # [T,L] complex
        var_f = pad(var_f, (self.L - 1, 0))  # [T+L-1]
        var_f_para = var_f.unfold(0, self.L, 1)  # [T,L] real

        Err_var_f_para = self.M_step_f_t_ErrVar_para(
            Obs_f_para, CTF_f_para, mu_f_para, var_f_para
        )  # m:T(12L+1) a:T(7L)
        Err_var_f_para = Err_var_f_para + NoiVar_f  # a:T
        Err_var_f_ret = Err_var_f_para[self.L : -self.L].real.mean(0)  # a:T

        return Err_var_f_ret

    @torch.no_grad()
    def M_step_f_CTF(self, Obs_f, CTF_f, mu_f, var_f, *args, **kwargs):
        """
        对每个频带估计CTF和噪声方差
        input:
            Obs_f: [T] complex
            CTF_f: [L] complex
            mu_f: [T] complex
            var_f: [T] real
        return:
            CTF_f_ret: [L] complex
            Err_var_f_ret: [] real
        Number of multiplications: 16/3L^3+(4T+4)L^2+4TL
        Number of addition: 2L^3+(2T+2)L^2+4T+3L
        """
        T = Obs_f.shape[0]
        Obs_f_para = Obs_f  # [T] complex
        CTF_f_para = CTF_f.unsqueeze(0).repeat(T, 1)  # [T,L] complex
        mu_f = pad(mu_f, (self.L - 1, 0))  # [T+L-1]
        mu_f_para = mu_f.unfold(0, self.L, 1)  # [T,L] complex
        var_f = pad(var_f, (self.L - 1, 0))  # [T+L-1]
        var_f_para = var_f.unfold(0, self.L, 1)  # [T,L] real

        CTF_f_PartA_para, CTF_f_PartB_para = self.M_step_f_t_CTF_para(
            Obs_f_para, CTF_f_para, mu_f_para, var_f_para
        )  # m:T(4L^2+4L) a:T(2L^2+3L)

        CTF_f_PartA_para = CTF_f_PartA_para[self.L : -self.L]
        CTF_f_PartB_para = CTF_f_PartB_para[self.L : -self.L]

        CTF_f_PartA = CTF_f_PartA_para.mean(0)  # a:2T
        CTF_f_PartB = CTF_f_PartB_para.mean(0)  # a:2T

        CTF_f_ret = torch.matmul(
            CTF_f_PartB, torch.inverse(CTF_f_PartA)
        ).squeeze()  # m:4*4/3*L^3+4L^2 a:2L^3+2L^2

        return CTF_f_ret

    @torch.no_grad()
    def cal_likeli_f_t(
        self, Obs_f_t, CTF_f, Mu_f_t, Var_f_t, ErrVar_f, *args, **kwargs
    ):
        """
        对每个时频点计算对数似然
        input:
            Obs_f_t: [] complex
            CTF_f: [L] complex
            Mu_f_t: [L] complex t-L:t
            Var_f_t: [L] real t-L:t
            ErrVar_f: [] real
        return:
            likeli_f_t: [] real
        """

        likeli_f_t = (
            -ErrVar_f.log()
            - (
                (
                    Obs_f_t.abs().pow(2)
                    - 2 * (Obs_f_t.conj() * ((CTF_f * Mu_f_t).sum())).real
                    + torch.matmul(
                        torch.matmul(
                            CTF_f.unsqueeze(0),
                            (
                                torch.matmul(
                                    Mu_f_t.unsqueeze(1), Mu_f_t.unsqueeze(0).conj()
                                )
                                + torch.diag_embed(Var_f_t)
                            ),
                        ),
                        CTF_f.unsqueeze(1).conj(),
                    ).squeeze()
                )
                / ErrVar_f
            ).real
        )

        return likeli_f_t

    @torch.no_grad()
    def cal_likeli_f(self, Obs_f, CTF_f, Mu_f, Var_f, ErrVar_f, *args, **kwargs):
        """
        对每个时频点计算对数似然
        input:
            Obs_f: [T] complex
            CTF_f: [L] complex
            Mu_f: [T] complex t-L:t
            Var_f: [T] real t-L:t
            ErrVar_f: [] real
        return:
            likeli_f: [] real
        """
        T = Obs_f.shape[0]
        Obs_f_para = Obs_f  # [T] complex

        CTF_f_para = CTF_f.unsqueeze(0).repeat(T, 1)  # [T,L] complex
        ErrVar_f_para = ErrVar_f.unsqueeze(0).repeat(T)
        Mu_f = pad(Mu_f, (self.L - 1, 0))  # [T+L-1]
        Mu_f_para = Mu_f.unfold(0, self.L, 1)  # [T,L] complex
        Var_f = pad(Var_f, (self.L - 1, 0))  # [T+L-1]
        Var_f_para = Var_f.unfold(0, self.L, 1)  # [T,L] real

        likeli_para = self.cal_likeli_f_t_para(
            Obs_f_para, CTF_f_para, Mu_f_para, Var_f_para, ErrVar_f_para
        )

        return likeli_para.mean()
