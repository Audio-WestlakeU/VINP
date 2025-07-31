# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: code for dataset and dataloader.

import numpy as np
from .base_dataset_torch import BaseDataset
from .utils import MyDistributedSampler
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torchaudio
import torchaudio.functional as F

from typing import List, Tuple, Optional, Union
import random
import soundfile as sf
from pathlib import Path
from typing import Optional, Union, Tuple


class MyDataset(BaseDataset):

    def __init__(
        self,
        src_pathlist_txt: str,
        rir_pathlist_txt: str,
        noise_pathlist_txt: str,
        snr_range: List[float] = [5, 20],
        seq_len: Union[None, float] = None,
        sr: int = 16000,
        shuffle: bool = False,
        noisy_proportion: float = 0.75,
        repeat=False,
        *args,
        **kwargs
    ) -> BaseDataset:
        """Noisy reverberate dataset for 1-ch recordings

        Args:
            src_pathlist_txt (str): paths of source speech in .txt file
            rir_pathlist_txt (str): paths of RIR in .txt file
            noise_pathlist_txt (str): paths of NOISE in .txt file
            snr_range (List[float], optional): range of SNR. Defaults to [20, 20].
            seq_len (Union[None, float], optional): sequence length. Defaults to None.
            sr (int, optional): target sample rate. Defaults to 16000.
            rir_type (str, optional): 'sim' or 'real'. Defaults to "real".
            shuffle (bool, optional): shuffle or not. Defaults to False.

        Returns:
            BaseDataset: Noisy reverberate dataset
        """

        super().__init__()

        self.sr = sr
        self.snr_range = snr_range
        self.noisy_proportion = noisy_proportion
        self.repeat = repeat
        # paths of the wavs
        self.source_pathlist = [
            line.rstrip("\n") for line in open(os.path.abspath(src_pathlist_txt), "r")
        ]
        self.source_pathlist = sorted(
            self.source_pathlist, key=lambda i: os.path.basename(i)
        )
        self.rir_pathlist = [
            line.rstrip("\n") for line in open(os.path.abspath(rir_pathlist_txt), "r")
        ]
        self.rir_pathlist = sorted(self.rir_pathlist, key=lambda i: os.path.basename(i))
        self.noise_pathlist = [
            line.rstrip("\n") for line in open(os.path.abspath(noise_pathlist_txt), "r")
        ]
        self.noise_pathlist = sorted(
            self.noise_pathlist, key=lambda i: os.path.basename(i)
        )

        self.seq_len = seq_len
        # self.rir_type = rir_type
        self.shuffle = shuffle
        self.length = len(self.source_pathlist)

    def gen_real_datapair(
        self, source: torch.Tensor, rir: torch.Tensor, dprir=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate real reverb-directpath data pairs

        Args:
            source (np.ndarray): source waveform
            rir (np.ndarray): rir waveform

        Returns:
            Tuple[np.ndarray,np.ndarray]: reverb and direct-path waveform
        """

        if not isinstance(dprir, torch.Tensor):
            dprir = torch.zeros_like(rir)
            dprirSNR = torch.zeros_like(rir)

            idx = torch.argmax(rir.abs())

            idx_beg = max(0, idx - int(self.sr * 0.0020))
            idx_end = idx + int(self.sr * 0.0020)

            idx_begSNR = max(0, idx - int(self.sr * 0.001))
            idx_endSNR = idx + int(self.sr * 0.05) + 1

            dprir[idx_beg:idx_end] = rir[idx_beg:idx_end]
            dprirSNR[idx_begSNR:idx_endSNR] = rir[idx_begSNR:idx_endSNR]

            reverb = F.fftconvolve(source, rir, mode="full")
            target = F.fftconvolve(source, dprir, mode="full")
            targetSNR = F.fftconvolve(source, dprirSNR, mode="full")

        else:
            reverb = F.fftconvolve(source, rir.squeeze(), mode="full")
            target = F.fftconvolve(source, dprir.squeeze(), mode="full")
            targetSNR = target

        return reverb, target, targetSNR

    def cut_or_pad(
        self,
        wav: torch.Tensor,
        seq_len: Union[float, None] = None,
        repeat: bool = False,
        rng=np.random.default_rng(),
    ) -> Tuple[torch.Tensor, int, int]:
        """cut or pad waveform

        Args:
            wav (np.ndarray): waveform
            seq_len (Union[float, None], optional): target sequence length. Defaults to None.
            repeat (bool, optional): repeat or not. Defaults to False.

        Returns:
            Tuple[np.ndarray,int,int]: waveform, start sample, end sample
        """
        T = wav.shape[-1]
        if seq_len != None:
            target_len = int(seq_len * self.sr)
        else:
            target_len = T

        if target_len < T:
            beg_idx = rng.integers(low=0, high=T - target_len)
            end_idx = beg_idx + target_len
            wav = wav[beg_idx:end_idx]
        elif target_len > T:
            if repeat == True:
                while T <= target_len:
                    wav = torch.cat(
                        [
                            wav,
                            torch.zeros(
                                [rng.integers(low=self.sr * 0.1, high=self.sr * 0.5)]
                            ),
                            wav,
                        ],
                        -1,
                    )
                    T = wav.shape[-1]
                beg_idx = rng.integers(low=0, high=T - target_len)
                end_idx = beg_idx + target_len
            else:
                pre_pad = rng.integers(low=0, high=target_len - T)
                post_pad = int(target_len - T - pre_pad)
                wav = torch.nn.functional.pad(
                    wav, (pre_pad, post_pad), mode="constant", value=0
                )
                beg_idx = 0
                end_idx = target_len
        else:
            beg_idx = 0
            end_idx = target_len

        return wav, beg_idx, end_idx

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self, index_seed: tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """get items

        Args:
            index_seed (tuple[int, int]): item index and random seed

        Returns:
            Tuple[np.ndarray,np.ndarray]: noisy and target waveform
        """

        index, seed = index_seed
        rng = np.random.default_rng(np.random.PCG64(seed + index))

        noise_exist = True if rng.uniform(0, 1) <= self.noisy_proportion else False

        source_fpath = self.source_pathlist[index]

        index_noise = rng.integers(low=0, high=len(self.noise_pathlist))
        noise_fpath = self.noise_pathlist[index_noise]

        snrdB = rng.uniform(self.snr_range[0], self.snr_range[1])

        source = self.load_wav(source_fpath, self.sr, None, rng)

        if source.abs().max() > 0:
            source /= source.abs().max()
        source, _, _ = self.cut_or_pad(source, self.seq_len, self.repeat, rng)

        if self.shuffle:
            rir_idx = rng.integers(low=0, high=len(self.rir_pathlist))
            rir_this = self.rir_pathlist[rir_idx]
        else:
            rir_this = self.rir_pathlist[index % len(self.rir_pathlist)]

        _, ext = os.path.splitext(rir_this)
        if ext == ".npz":
            rir_dict = np.load(rir_this)
            sr_rir = rir_dict["fs"]
            rir = torch.Tensor(rir_dict["rir"][0])
            dprir = torch.Tensor(rir_dict["rir_dp"][0])
            scale_rir = rir.abs().max()
            rir /= scale_rir
            dprir /= scale_rir
            reverb, target, targetSNR = self.gen_real_datapair(source, rir, dprir)

        else:
            try:
                rir = self.load_wav(rir_this, self.sr, None, rng)
            except:
                print(rir_this)
            scale_rir = rir.abs().max()
            rir /= scale_rir

            reverb, target, targetSNR = self.gen_real_datapair(source, rir, None)

        target, beg_sample, end_sample = self.cut_or_pad(
            target, self.seq_len, False, rng
        )
        reverb = reverb[beg_sample:end_sample]
        targetSNR = targetSNR[beg_sample:end_sample]

        if noise_exist:

            noise = self.load_wav(noise_fpath, self.sr, None, rng)
            noise, _, _ = self.cut_or_pad(
                noise, (reverb.shape[-1] + 1) / self.sr, True, rng
            )
            noise = noise[..., : reverb.shape[-1]]

            dp_power = targetSNR.pow(2).mean()
            noise_power = noise.pow(2).mean()

            Msnr = (10 ** (-snrdB / 10) * dp_power / (noise_power + self.eps)).sqrt()
            noise = noise * Msnr

            assert reverb.shape == noise.shape, print(reverb.shape, noise.shape)

            reverb = reverb + noise

        scale = reverb.abs().max()

        target = target / (scale + self.eps)
        reverb = reverb / (scale + self.eps)

        return reverb.unsqueeze(0), target.unsqueeze(0), source_fpath


class MyDataloader(DataLoader):

    def __init__(
        self,
        src_pathlist: Tuple[str, str, str],
        rir_pathlist: Tuple[str, str, str],
        noise_pathlist: Tuple[str, str, str],
        snr_range: Tuple[float, float],
        seq_lenlist: Tuple[Union[float, None], Union[float, None], Union[float, None]],
        batchsize: Tuple[int, int, int],
        sr: int,
        num_workers: int,
        seeds: Tuple[Union[int, None], Union[int, None], Union[int, None]] = [
            None,
            None,
            None,
        ],
        pin_memory: bool = True,
        prefetch_factor: Union[None, int] = None,
        persistent_workers: Union[None, bool] = None,
        rank: int = 0,
        noisy_proportion: float = 0.75,
        *args,
        **kwargs
    ) -> DataLoader:
        """my dataloader

        Args:
            src_pathlist (Tuple[str, str, str]): source path .txt for training, validating and testing
            rir_pathlist (Tuple[str, str, str]): RIR path .txt for training, validating and testing
            noise_pathlist (Tuple[str, str, str]): Noise path .txt for training, validating and testing
            snr_range (Tuple[float, float]): SNR range for training, validating and testing
            seq_lenlist (Tuple[Union[float, None], Union[float, None], Union[float, None]]): sequence length for training, validating and testing
            batchsize (Tuple[int, int, int]): batch size for training, validating and testing
            sr (int): sampling rate
            num_workers (int): _description_
            seeds (Tuple[Union[int, None], Union[int, None], Union[int, None]], optional): random seeds. Defaults to [ None, None, None, ].
            pin_memory (bool, optional): _description_. Defaults to True.
            prefetch_factor (Union[None, int], optional): _description_. Defaults to None.
            persistent_workers (Union[None, bool], optional): _description_. Defaults to None.
            rank (int, optional): device rank, 0 for validation. Defaults to 0.

        Returns:
            DataLoader
        """

        self.src_pathlist = src_pathlist
        self.rir_pathlist = rir_pathlist
        self.noise_pathlist = noise_pathlist
        self.snr_range = snr_range
        self.noisy_proportion = noisy_proportion
        self.seq_lenlist = seq_lenlist
        self.batchsize = batchsize
        self.sr = sr
        self.num_workers = num_workers
        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.rank = rank
        self.train_dataloader = self.build_train_dataloader()
        self.valid_dataloader = self.build_valid_dataloader()

    def construct_dataloader(
        self,
        src_path_txt,
        rir_path_txt,
        noise_path_txt,
        snr_range,
        seq_len,
        batchsize,
        seed,
        src_shuffle,
        rir_shuffle,
        repeat,
        sampler: bool,
    ):
        ds = MyDataset(
            src_path_txt,
            rir_path_txt,
            noise_path_txt,
            snr_range,
            seq_len,
            self.sr,
            rir_shuffle,
            self.noisy_proportion,
            repeat,
        )

        return DataLoader(
            ds,
            sampler=(
                MyDistributedSampler(ds, seed=seed, shuffle=src_shuffle, rank=self.rank)
                if sampler
                else MyDistributedSampler(
                    ds, seed=seed, shuffle=src_shuffle, rank=0, num_replicas=1
                )
            ),  #
            batch_size=batchsize,  #
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def build_train_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            src_path_txt=self.src_pathlist[0],
            rir_path_txt=self.rir_pathlist[0],
            noise_path_txt=self.noise_pathlist[0],
            snr_range=self.snr_range,
            seq_len=self.seq_lenlist[0],
            batchsize=self.batchsize[0],
            seed=self.seeds[0],
            src_shuffle=True,
            rir_shuffle=True,
            sampler=True,
            repeat=True,
        )

    def build_valid_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            src_path_txt=self.src_pathlist[1],
            rir_path_txt=self.rir_pathlist[1],
            noise_path_txt=self.noise_pathlist[1],
            snr_range=self.snr_range,
            seq_len=self.seq_lenlist[1],
            batchsize=self.batchsize[1],
            seed=self.seeds[1],
            src_shuffle=False,
            rir_shuffle=False,
            sampler=True,
            repeat=False,
        )

    def build_test_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            src_path_txt=self.src_pathlist[2],
            rir_path_txt=self.rir_pathlist[2],
            noise_path_txt=self.noise_pathlist[2],
            snr_range=self.snr_range,
            seq_len=self.seq_lenlist[2],
            batchsize=self.batchsize[2],
            seed=self.seeds[2],
            src_shuffle=False,
            rir_shuffle=False,
            sampler=False,
            repeat=False,
        )

    def build_predict_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            src_path_txt=self.src_pathlist[2],
            rir_path_txt=self.rir_pathlist[2],
            noise_path_txt=self.noise_pathlist[2],
            snr_range=self.snr_range,
            seq_len=None,
            batchsize=1,
            seed=0,
            src_shuffle=False,
            rir_shuffle=False,
            repeat=False,
        )


if __name__ == "__main__":
    dataset = MyDataset(
        "config/path_src_train.txt",
        "config/path_rir_train_sim.txt",
        "config/path_noise_train.txt",
        [5, 20],
        None,
        16000,
        shuffle=False,
        noisy_proportion=1,
    )
    out_path = "/DATASET/fixed_200h_final/train"
    out_reverb_path = os.path.join(out_path, "noisy")
    out_dp_path = os.path.join(out_path, "clean")
    os.makedirs(out_reverb_path, exist_ok=True)
    os.makedirs(out_dp_path, exist_ok=True)

    import soundfile as sf
    from tqdm import tqdm

    count = 0
    for i in tqdm(range(dataset.__len__())):

        reverb, target, filename = dataset.__getitem__((i, i))

        if np.max(np.abs(reverb.numpy())) > 1:
            scale = np.max(np.abs(reverb.numpy())) + 1e-32
        else:
            scale = 1
        reverb = reverb.numpy() / scale
        target = target.numpy() / scale

        count += 1
        if filename[-4:] == "flac":
            sf.write(
                os.path.join(out_reverb_path, os.path.basename(filename)[:-4] + "flac"),
                reverb.swapaxes(0, 1),
                16000,
                "PCM_16",
            )
            sf.write(
                os.path.join(out_dp_path, os.path.basename(filename)[:-4] + "flac"),
                target.swapaxes(0, 1),
                16000,
                "PCM_16",
            )
        elif filename[-3:] == "wav":
            sf.write(
                os.path.join(out_reverb_path, os.path.basename(filename)[:-3] + "flac"),
                reverb.swapaxes(0, 1),
                16000,
                "PCM_16",
            )
            sf.write(
                os.path.join(out_dp_path, os.path.basename(filename)[:-3] + "flac"),
                target.swapaxes(0, 1),
                16000,
                "PCM_16",
            )
