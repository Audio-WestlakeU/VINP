# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: code for generating SimACE testset.

import numpy as np
from base_dataset_torch import BaseDataset
from utils import MyDistributedSampler
import os
import torch
from scipy import signal
from torch.utils.data import DataLoader, DistributedSampler
import torchaudio
import torchaudio.functional as F
import soundfile as sf
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
import random
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Union, Tuple
import json
from jsonargparse import ArgumentParser

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
        *args, **kwargs
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

            
            reverb=F.fftconvolve(source,rir,mode="full")
            target=F.fftconvolve(source,dprir,mode="full")
            targetSNR = F.fftconvolve(source,dprirSNR,mode="full")
            
        else:
            reverb=F.fftconvolve(source,rir.squeeze(),mode="full")
            target=F.fftconvolve(source,dprir.squeeze(),mode="full")
            targetSNR = target

        return reverb, target,targetSNR

    def cut_or_pad(
        self,
        wav: torch.Tensor,
        seq_len: Union[float, None] = None,
        repeat: bool = False,
        rng = np.random.default_rng()
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
                    wav = torch.cat([wav,torch.zeros([rng.integers(low=self.sr*0.1,high=self.sr*0.3)]), wav], -1)
                    T = wav.shape[-1]
                beg_idx = rng.integers(low=0,  high=T - target_len)
                end_idx = beg_idx + target_len
            else:
                pre_pad = rng.integers(low=0, high=target_len - T)
                post_pad = int(target_len - T - pre_pad)
                wav = torch.nn.functional.pad(wav,(pre_pad, post_pad),mode='constant', value=0)
                beg_idx = 0
                end_idx = target_len
        else:
            beg_idx = 0
            end_idx = target_len

        return wav, beg_idx, end_idx

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index_seed: tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """get items

        Args:
            index_seed (tuple[int, int]): item index and random seed

        Returns:
            Tuple[np.ndarray,np.ndarray]: noisy and target waveform
        """

        index, seed = index_seed
        rng = np.random.default_rng(np.random.PCG64(seed + index))
        
        noise_exist = True if rng.uniform(0, 1)<=self.noisy_proportion else False
        
        source_fpath = self.source_pathlist[index]
        
        index_noise = rng.integers(low=0, high=len(self.noise_pathlist))
        noise_fpath = self.noise_pathlist[index_noise]

        snrdB = rng.uniform(self.snr_range[0], self.snr_range[1])

        source = self.load_wav(source_fpath, self.sr, None,rng)

        if source.abs().max()>0:
            source/= source.abs().max()
        source, _, _ = self.cut_or_pad(source, self.seq_len, True,rng)



        if self.shuffle:
            rir_idx = rng.integers(low=0, high=len(self.rir_pathlist))
            rir_this = self.rir_pathlist[rir_idx]
        else:
            rir_this = self.rir_pathlist[index % len(self.rir_pathlist)]

        _, ext = os.path.splitext(rir_this)
        if ext=='.npz':
            rir_dict = np.load(rir_this)
            sr_rir = rir_dict["fs"]
            rir = torch.Tensor(rir_dict["rir"][0])
            dprir = torch.Tensor(rir_dict["rir_dp"][0])
            scale_rir = rir.abs().max()
            rir/=scale_rir
            dprir/=scale_rir
            reverb, target, targetSNR = self.gen_real_datapair(source, rir,dprir)
            
        else:
            try:
                rir = self.load_wav(rir_this, self.sr, None,rng)
                if '403a' in rir_this:
                    T60 = 1.22
                elif '502' in rir_this:
                    T60 = 0.332
                elif '503' in rir_this:
                    T60 = 0.437
                elif '508' in rir_this:
                    T60 = 0.638
                elif '611' in rir_this:
                    T60 = 0.371
                elif '803' in rir_this:
                    T60 = 0.39
                else:
                    T60 = 0.646
                    
                dp_start_sample = int(max(np.argmax(np.abs(rir))-0.0025*self.sr,0))
                dp_stop_sample = int(np.argmax(np.abs(rir))+0.0025*self.sr)

                rir_power = (rir**2).sum()
                dp_power = (rir[dp_start_sample:dp_stop_sample]**2).sum()
                DRR = 10*(dp_power/(rir_power-dp_power)).log10().item()
            except:
                print(rir_this)
            scale_rir = rir.abs().max()
            rir/=scale_rir

            reverb, target, targetSNR = self.gen_real_datapair(source, rir,None)
            
        


        target, beg_sample, end_sample = self.cut_or_pad(target, self.seq_len, False,rng)
        reverb = reverb[beg_sample:end_sample]
        targetSNR = targetSNR[beg_sample:end_sample]
        

        
        if noise_exist:

            noise = self.load_wav(noise_fpath, self.sr, None,rng)
            noise, _, _ = self.cut_or_pad(
                noise, (reverb.shape[-1]+1)/self.sr, True,rng
            )
            
            noise = noise[...,:reverb.shape[-1]]


            dp_power = targetSNR.pow(2).mean()
            noise_power = noise.pow(2).mean()

            Msnr = (10 ** (-snrdB / 10) * dp_power / (noise_power + self.eps)).sqrt()
            noise = noise * Msnr

            
            assert reverb.shape == noise.shape, print(reverb.shape,noise.shape)
            
            reverb = reverb + noise
            

        scale = reverb.abs().max()
        
        target = target / (scale + self.eps)
        reverb = reverb / (scale + self.eps)

        return reverb.unsqueeze(0), target.unsqueeze(0),source_fpath,T60,DRR



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--src_pathlist_txt", required=True, type=str
    )
    parser.add_argument(
        "--rir_pathlist_txt", required=True, type=str
    )
    parser.add_argument(
        "--noise_pathlist_txt", required=True, type=str
    )
    parser.add_argument(
        "--snr_range", required=True, type=list
    )
    parser.add_argument(
        "--seq_len", required=False, default=None
    )
    parser.add_argument(
        "--sr", required=False, type=int, default=16000
    )
    parser.add_argument(
        "--shuffle", required=False, type=bool, default=False
    )
    parser.add_argument(
        "--noisy_proportion", required=False, type=float, default=1
    )
    parser.add_argument(
        "--out_path", required=True, type=str
    )
    args = parser.parse_args()
    
    dataset = MyDataset(**args)
    
    out_reverb_path = os.path.join(args.out_path, "noisy")
    out_dp_path = os.path.join(args.out_path, "clean")
    os.makedirs(out_reverb_path, exist_ok=True)
    os.makedirs(out_dp_path, exist_ok=True)

    T60_log = {}
    DRR_log = {}
    count = 0
    for i in tqdm(range(dataset.__len__())):

        reverb, target,filename,T60,DRR = dataset.__getitem__((i, i))
        T60_log[os.path.basename(filename)] = T60
        DRR_log[os.path.basename(filename)] = DRR

        if np.max(np.abs(reverb.numpy()))>1:
            scale = np.max(np.abs(reverb.numpy())) + 1e-32
        else:
            scale = 1
        reverb = reverb.numpy() / scale
        target = target.numpy() / scale

        count += 1
        if filename[-4:]=='flac':
            sf.write(
                os.path.join(out_reverb_path, os.path.basename(filename)[:-4]+'flac'),
                reverb.swapaxes(0, 1),
                16000,
                "PCM_16",
            )
            sf.write(
                os.path.join(out_dp_path, os.path.basename(filename)[:-4]+'flac'),
                target.swapaxes(0, 1),
                16000,
                "PCM_16",
            )
        elif filename[-3:]=='wav':
            sf.write(
                os.path.join(out_reverb_path, os.path.basename(filename)[:-3]+'flac'),
                reverb.swapaxes(0, 1),
                16000,
                "PCM_16",
            )
            sf.write(
                os.path.join(out_dp_path, os.path.basename(filename)[:-3]+'flac'),
                target.swapaxes(0, 1),
                16000,
                "PCM_16",
            )
            
    with open(os.path.join(os.path.join(args.out_path,'_T60s.json')),'w',encoding='utf-8') as f:
        json.dump(T60_log, f, ensure_ascii=False, indent=4)
    with open(os.path.join(os.path.join(args.out_path,'_DRRs.json')),'w',encoding='utf-8') as f:
        json.dump(DRR_log, f, ensure_ascii=False, indent=4)
        
    '''
    usage: python gen_SimACE_testset.py --[keyword] [arg]
    '''