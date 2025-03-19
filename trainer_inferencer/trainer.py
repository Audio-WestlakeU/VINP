# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: trainer.

import torch
import numpy as np
from torch.cuda.amp import autocast
from tqdm import tqdm
import torch.distributed as dist
from .base_trainer import BaseTrainer
from .utils import plot_spectrogram, initialize_module
import matplotlib.pyplot as plt
from model.lossF import IS_loss, log_MSE_loss

lossF_IS = IS_loss()
lossF_logMSE = log_MSE_loss()


class Trainer(BaseTrainer):
    def __init__(
        self,
        dist,
        rank,
        config,
        resume: bool,
        model,
        loss_func,
        optimizer,
        scheduler,
        train_dataloader,
        valid_dataloader,
        start_ckpt,
        *args,
        **kwargs,
    ):
        super().__init__(
            dist,
            rank,
            config,
            resume,
            model,
            loss_func,
            optimizer,
            scheduler,
            start_ckpt,
        )

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.EM_algo = initialize_module(
            config["EM_algo"]["path"], args=config["EM_algo"]["args"]
        )

    def _train_epoch(self, epoch):

        loss_total = 0.0
        self.optimizer.zero_grad()
        for index, (rev_wav, dp_wav, fpath) in (
            enumerate(tqdm(self.train_dataloader, desc="Training"))
            if self.rank == 0
            else enumerate(self.train_dataloader)
        ):

            rev_wav = rev_wav.to(self.rank)
            dp_wav = dp_wav.to(self.rank)
            assert (
                not torch.isnan(rev_wav).any() and not torch.isnan(dp_wav).any()
            ), "NaN in input or output"

            input_absSpec, _ = self.transformfunc.stft(rev_wav, output_type="mag_phase")
            input_feature = self.transformfunc.preprocess(input_absSpec)  # [B,1,F,T]

            target_absSpec, _ = self.transformfunc.stft(dp_wav, output_type="mag_phase")

            with autocast(enabled=self.use_amp, dtype=torch.float16):
                output_feature = self.model(input_feature)
                output_absSpec = self.transformfunc.postprocess(output_feature)
                loss = self.loss_func(output_absSpec, target_absSpec)

            self.scaler.scale(loss).backward(retain_graph=True)
            gradient_norm = self.get_gradient_norm()

            loss_total = loss_total + loss.item()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm_value
            )

            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if torch.isnan(param.grad).any():
                    has_nan_grad = True

            if not has_nan_grad:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                print(f"Check source file {fpath}")
            self.optimizer.zero_grad()

            if (index / self.grad_accumulation_steps) % 100 == 0:
                if self.rank == 0:
                    input_absSpec_exp = input_absSpec[0, 0].detach().cpu().numpy()
                    output_absSpec_exp = output_absSpec[0, 0].detach().cpu().numpy()
                    target_absSpec_exp = target_absSpec[0, 0].detach().cpu().numpy()
                    self.writer.add_figure(
                        f"Train/Input_feature_image",
                        plot_spectrogram(input_absSpec_exp),
                    )
                    self.writer.add_figure(
                        f"Train/Output_feature_image",
                        plot_spectrogram(output_absSpec_exp),
                    )
                    self.writer.add_figure(
                        f"Train/Target_feature_image",
                        plot_spectrogram(target_absSpec_exp),
                    )
                    self.writer.add_scalar(
                        f"Train_step/Loss",
                        loss.item() * self.grad_accumulation_steps,
                        (index / self.grad_accumulation_steps)
                        + (epoch - 1) * len(self.train_dataloader),
                    )
                    self.writer.add_scalar(
                        f"Train_step/Grad_L2",
                        gradient_norm,
                        (index / self.grad_accumulation_steps)
                        + (epoch - 1) * len(self.train_dataloader),
                    )
        if self.rank == 0:
            self.writer.add_scalar(
                f"Train/Loss",
                loss_total / len(self.train_dataloader),
                epoch,
            )

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        loss_IS_total = 0.0
        loss_logMSE_total = 0.0

        for index, (rev_wav, dp_wav, fpath) in (
            enumerate(tqdm(self.valid_dataloader, desc="Validating"))
            if self.rank == 0
            else enumerate(self.valid_dataloader)
        ):
            rev_wav = rev_wav.to(self.rank)
            dp_wav = dp_wav.to(self.rank)

            input_absSpec, _ = self.transformfunc.stft(rev_wav, output_type="mag_phase")
            input_feature = self.transformfunc.preprocess(input_absSpec)  # [B,1,F,T]

            target_absSpec, _ = self.transformfunc.stft(dp_wav, output_type="mag_phase")
            with autocast(enabled=self.use_amp):
                if self.model_ema:
                    output_feature = self.model_ema(input_feature)
                else:
                    output_feature = self.model(input_feature)
                output_absSpec = self.transformfunc.postprocess(output_feature)
                loss = (
                    self.loss_func(output_absSpec, target_absSpec)
                    / self.grad_accumulation_steps
                )
                loss_IS = (
                    lossF_IS(output_absSpec, target_absSpec)
                    / self.grad_accumulation_steps
                )
                loss_logMSE = (
                    lossF_logMSE(output_absSpec, target_absSpec)
                    / self.grad_accumulation_steps
                )

            # 同步 loss
            dist.reduce(
                loss, dst=0
            )  # 将各个卡上的 loss 汇总到主进程（这里假设rank为0是主进程，可根据实际情况调整）
            dist.reduce(loss_IS, dst=0)
            dist.reduce(loss_logMSE, dst=0)
            if self.rank == 0:  # 只在主进程进行后续的累加等操作，避免重复计算
                loss_total = loss_total + loss.item() * self.grad_accumulation_steps
                loss_IS_total = (
                    loss_IS_total + loss_IS.item() * self.grad_accumulation_steps
                )
                loss_logMSE_total = (
                    loss_logMSE_total
                    + loss_logMSE.item() * self.grad_accumulation_steps
                )
                if index == 0:
                    for iexp in range(1):
                        input_absSpec_exp = (
                            input_absSpec[iexp, 0].detach().cpu().numpy()
                        )
                        output_absSpec_exp = (
                            output_absSpec[iexp, 0].detach().cpu().numpy()
                        )
                        target_absSpec_exp = (
                            target_absSpec[iexp, 0].detach().cpu().numpy()
                        )

                        self.writer.add_figure(
                            f"Valid/Input_feature_image{iexp}",
                            plot_spectrogram(input_absSpec_exp),
                        )
                        self.writer.add_figure(
                            f"Valid/Output_feature_image{iexp}",
                            plot_spectrogram(output_absSpec_exp),
                        )
                        self.writer.add_figure(
                            f"Valid/Target_feature_image{iexp}",
                            plot_spectrogram(target_absSpec_exp),
                        )

                self.writer.add_scalar(
                    f"Valid/Loss",
                    loss_total / len(self.valid_dataloader) / dist.get_world_size(),
                    epoch,
                )
                self.writer.add_scalar(
                    f"Valid/LossIS",
                    loss_IS_total / len(self.valid_dataloader) / dist.get_world_size(),
                    epoch,
                )
                self.writer.add_scalar(
                    f"Valid/LossLogMSE",
                    loss_logMSE_total
                    / len(self.valid_dataloader)
                    / dist.get_world_size(),
                    epoch,
                )
        if self.rank == 0:  # 最终只让主进程返回验证的平均 loss
            return loss_total / len(self.valid_dataloader) / dist.get_world_size()
        else:
            return None
