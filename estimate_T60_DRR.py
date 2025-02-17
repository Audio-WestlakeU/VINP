# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: calculate RT60 and DRR given RIRs.

from glob import glob
from tqdm import tqdm
import os
import torch
import numpy as np
from acoustics.feature import (
    load_wav,
    norm_amplitude,
)
from jsonargparse import ArgumentParser
import json
import scipy
import matplotlib.pyplot as plt


def save_config_to_file(args, file_path):
    with open(file_path, "w") as json_file:
        json.dump(args.__dict__, json_file, indent=4)


def cal_T60(RIR, sr, savepath):
    RIR = RIR[np.argmax(np.abs(RIR)) :]
    EDC = np.zeros_like(RIR)
    power = np.abs(RIR) ** 2
    power_flip = np.flipud(power)
    edc = np.zeros_like(power)
    fow_pow = 0
    for isample in range(power_flip.shape[0]):
        fow_pow = fow_pow + power_flip[isample]
        edc[isample] = fow_pow
    EDC = 10 * np.log10(np.flipud(edc) / power.sum())
    taxis = np.arange(0, EDC.shape[0], 1) / sr
    r_final = 0
    for beg_sample in range(int(sr * 0.02), int(sr * 0.05)):
        end_sample = beg_sample
        while EDC[end_sample] > EDC[beg_sample] - 5:
            end_sample = end_sample + 1
        line_fit = EDC[beg_sample:end_sample]
        taxis_fit = taxis[beg_sample:end_sample]
        k, b, r, _, _ = scipy.stats.linregress(taxis_fit, line_fit)
        if r < r_final:
            beg_sample_final = beg_sample
            end_sample_final = end_sample
            r_final = r
            k_final = k
            b_final = b

    y = k_final * taxis[beg_sample_final:end_sample_final] + b_final

    RT60 = -60 / k
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.figure()
    plt.plot(taxis, EDC)
    plt.plot(taxis[beg_sample_final:end_sample_final], y, label="fit")
    plt.xlim(0, 0.3)
    plt.ylim(-40, 1)
    plt.legend()
    plt.title(f"{RT60}")
    plt.savefig(savepath)
    plt.close()

    return RT60


def est_T60(rir_path: str, sr=16000):
    fpath_rir = sorted(
        glob("{}/**/*{}".format(rir_path, ".flac"), recursive=True)
    ) + sorted(glob("{}/**/*{}".format(rir_path, ".wav"), recursive=True))

    T60_log = {}
    DRR_log = {}
    for fpath_input_n in tqdm(fpath_rir):
        basename_input_n = os.path.basename(fpath_input_n)

        fpath_out_edc = os.path.join(
            rir_path, "edc", basename_input_n.split(".")[0] + ".png"
        )

        rir = load_wav(fpath_input_n, sr)
        rir, _ = norm_amplitude(rir)

        T60 = cal_T60(rir, sr, fpath_out_edc)
        T60_log[basename_input_n] = T60

        dp_start_sample = int(max(np.argmax(np.abs(rir)) - 0.0025 * sr, 0))
        dp_stop_sample = int(np.argmax(np.abs(rir)) + 0.0025 * sr)

        rir_power = (rir**2).sum()
        dp_power = (rir[dp_start_sample:dp_stop_sample] ** 2).sum()
        DRR = 10 * (dp_power / (rir_power - dp_power)).log10().item()
        DRR_log[basename_input_n] = DRR

        print(T60, DRR)

    with open(
        os.path.join(os.path.join(rir_path, "_T60s.json")), "w", encoding="utf-8"
    ) as f:
        json.dump(T60_log, f, ensure_ascii=False, indent=4)
    with open(
        os.path.join(os.path.join(rir_path, "_DRRs.json")), "w", encoding="utf-8"
    ) as f:
        json.dump(DRR_log, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = ArgumentParser(
        description="code for generating wavs by PengyuWang @ Westlake University"
    )
    parser.add_argument(
        "-i", "--input_path", required=True, type=str, help="input path"
    )

    args = parser.parse_args()

    est_T60(args["input_path"])

    """
    usage: python estimate_T60_DRR.py -i [RIR dirpath]
    """
