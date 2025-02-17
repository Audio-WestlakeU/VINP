# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn

import torch
import os
import torchaudio
from pathlib import Path
from typing import Tuple


def load_wav(filename: str, sr_target: int = 16000) -> torch.Tensor:
    """load waveform for testing

    Args:
        filename (str): waveform path
        sr_target (int, optional): target sampling rate. Defaults to 16000.

    Returns:
        np.ndarray: loaded waveform
    """
    wav_path = Path(filename)
    assert wav_path.exists(), f"'{wav_path}' does not exist"
    wav_info = torchaudio.info(wav_path, backend="soundfile")
    sr_raw = wav_info.sample_rate
    n_ch = wav_info.num_channels
    num_frames = wav_info.num_frames
    assert sr_raw == sr_target, "sample rate not match"
    assert n_ch == 1, "1-ch supported"
    wav, _ = torchaudio.load(wav_path, channels_first=True)
    wav = wav.squeeze()

    return wav


def save_wav(wav: torch.Tensor, filename: str, sr: int = 16000) -> None:
    """save waveform for testing

    Args:
        wav (np.ndarray): enhanced waveform
        filename (str): save filepath
        sr (int, optional): sampling rate. Defaults to 16000.

    Returns:
        None
    """

    filename = Path(filename)

    os.makedirs(filename.parent, exist_ok=True)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(filename, wav.cpu(), sr, backend="soundfile")

    return None


def norm_amplitude(wav: torch.Tensor, eps: float = 1e-32) -> Tuple[torch.Tensor, float]:
    """Normalize waveform

    Args:
        wav (np.ndarray): waveform
        eps (float, optional): epsilon. Defaults to 1e-32.

    Returns:
        Tuple[np.ndarray, float]: [normalized waveform, original scale]
    """
    ori_scale = wav.abs().max() + eps

    return wav / ori_scale, ori_scale


class transforms:

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop_len: int,
        win_type: str,
        win_len: int,
        *args,
        **kwargs,
    ):
        """transformer

        Args:
            sr (int): sampling rate
            n_fft (int): fft points
            hop_len (int): hop length
            win_type (str): window type, hann, hamming, sqrthann, blackman supported
            win_len (int): window length
            # n_mel (int): number of mel filters
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        assert win_len % hop_len == 0, "win_len should be integral multiple of hop_len"
        if win_type.upper() == "HANN":
            win = torch.hann_window(win_len)
        elif win_type.upper() == "HAMMING":
            win = torch.hamming_window(win_len, periodic=False)
        elif win_type.upper() == "SQRTHANN":
            win = torch.hann_window(win_len).pow(0.5)
        elif win_type.upper() == "BLACKMAN":
            win = torch.blackman_window(win_len, periodic=False)

        self.awin = win
        self.swin = win

    def stft(
        self, seq: torch.Tensor, output_type: str = "complex"
    ) -> Tuple[torch.Tensor]:
        """STFT

        Args:
            seq (torch.Tensor): sequence
            output_type (str, optional): output type, complex, mag_phase, real_imag supported. Defaults to "complex".

        Returns:
            Tuple[torch.Tensor]: spectrogram
        """

        assert output_type.lower() in ["complex", "mag_phase", "real_imag"]
        if seq.ndim == 1:
            T = seq.shape[0]
            P = 1
        elif seq.ndim == 2:
            bs, T = seq.shape
            P = bs
        elif seq.ndim == 3:
            bs, C, T = seq.shape
            P = int(bs * C)
        else:
            raise ValueError("stft input should be less than 3D")

        seq_reshape = seq.reshape(P, T)
        spec_reshape = torch.stft(
            seq_reshape,
            self.n_fft,
            self.hop_len,
            self.win_len,
            window=self.awin.to(seq.device),
            return_complex=True,
        )

        F, nframe = spec_reshape.shape[-2:]

        if seq.ndim == 1:
            spec = spec_reshape.reshape([F, nframe])
            spec = spec[:, :]
        elif seq.ndim == 2:
            spec = spec_reshape.reshape([bs, F, nframe])
            spec = spec[:, :, :]
        elif seq.ndim == 3:
            spec = spec_reshape.reshape([bs, C, F, nframe])
            spec = spec[:, :, :, :]

        if output_type.upper() == "COMPLEX":
            return spec.contiguous()
        elif output_type.upper() == "MAG_PHASE":
            return spec.abs().contiguous(), spec.angle().contiguous()
        elif output_type.upper() == "REAL_IMAG":
            return spec.real.contiguous(), spec.imag.contiguous()

    def istft(
        self,
        spec: torch.Tensor,
        input_type: str = "complex",
        wav_len: Tuple[float, None] = None,
    ) -> torch.Tensor:
        """inverse STFT

        Args:
            spec (torch.Tensor): spectrogram
            input_type (str, optional): inpyt type. Defaults to "complex".
            wav_len (Tuple[float,None], optional): output length. Defaults to None.

        Returns:
            torch.Tensor: waveform
        """

        assert input_type.lower() in ["complex", "mag_phase", "real_imag"]
        if input_type.upper() == "COMPLEX":
            pass
        elif input_type.upper() == "MAG_PHASE":
            spec = spec[0] * torch.exp(1j * spec[1])
        elif input_type.upper() == "REAL_IMAG":
            spec = spec[0] + 1j * spec[1]

        if spec.ndim == 2:
            F, N = spec.shape
            P = 1
        elif spec.ndim == 3:
            bs, F, N = spec.shape
            P = bs
        elif spec.ndim == 4:
            bs, C, F, N = spec.shape
            P = int(bs * C)

        spec_reshape = spec.reshape([P, F, N])

        wav_reshape = torch.istft(
            spec_reshape,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.swin.to(spec.device),
            onesided=True,
        )

        T = wav_reshape.shape[-1]
        if wav_len:
            wav_reshape = wav_reshape[..., : min(T, wav_len)]
            T = wav_reshape.shape[-1]

        if spec.ndim == 2:
            F, N = spec.shape
            wav = wav_reshape.reshape([T])
        elif spec.ndim == 3:
            bs, F, N = spec.shape
            wav = wav_reshape.reshape([bs, T])
        elif spec.ndim == 4:
            bs, C, F, N = spec.shape
            wav = wav_reshape.reshape([bs, C, T])

        return wav.contiguous()

    def preprocess(self, input: torch.Tensor) -> torch.Tensor:
        """
        输入网络前的处理，包括幅度变换、维度变换等
        """
        output = (input + 1e-8).log10()

        return output.contiguous()

    def postprocess(self, input: torch.Tensor) -> torch.Tensor:
        """
        网络输出后的处理，包括幅度变换、维度变换等
        """
        output = 10 ** (input.clamp(-8, 6))
        return output.contiguous()
