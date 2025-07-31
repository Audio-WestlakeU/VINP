import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union
import numpy as np


class BaseDataset(Dataset):

    def __init__(self) -> None:
        """initialization"""
        super().__init__()
        self.eps = 1e-32

    def load_wav(
        self,
        wav_path: str,
        sr_target: int = 16000,
        target_len: Union[float, None] = None,
        rng = np.random.default_rng(),
    ) -> torch.Tensor:
        """load waveform / 读取波形

        Args:
            wav_path (str): path of waveform
            sr_target (int, optional): target sampling rate. Defaults to 16000.
            target_len (Union[float, None], optional): target length of waveform, None for no-cutting and no-padding. Defaults to None.

        Returns:
            np.ndarray: loaded waveform
        """

        # wav_path = Path(wav_path)
        # assert wav_path.exists(), f"'{wav_path}' does not exist"
        
        wav_info=torchaudio.info(wav_path, backend="soundfile")
        sr_raw = wav_info.sample_rate
        n_ch = wav_info.num_channels
        num_frames = wav_info.num_frames

        assert sr_raw == sr_target, 'sample rate not match'
        
        current_duration = num_frames/sr_target
        

        if target_len!=None and target_len<current_duration:
            
            sample_beg = rng.integers(low=0, high=(current_duration - target_len) * sr_target)
            sample_end = int(sample_beg+target_len * sr_target)
            assert sample_beg+int(target_len * sr_target)<=num_frames, f"{sample_beg},{int(target_len * sr_target)},{num_frames}"

            wav,_ = torchaudio.load(wav_path,channels_first=True,frame_offset=sample_beg,num_frames=int(target_len * sr_target),backend='sox')

        else:
            wav,_ = torchaudio.load(wav_path,channels_first=True)



        wavform = wav[rng.integers(0,n_ch),:]

        return wavform
