# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: resample waveforms.

import os
import soundfile as sf
import librosa
from glob import glob
import numpy as np
from jsonargparse import ArgumentParser


def downsample_and_save(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    rir_dir = input_dir

    rir_pathlist = glob(os.path.join(rir_dir, "**", "*_RIR.wav"), recursive=True)

    for rir_fpath in rir_pathlist:
        rir, sr = sf.read(rir_fpath)
        rir = librosa.resample(rir, orig_sr=sr, target_sr=16000)

        sf.write(
            os.path.join(output_dir, os.path.basename(rir_fpath)),
            rir / np.max(np.abs(rir)),
            16000,
        )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input_dir", required=True, type=str, help="input dirpath"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str, help="output dirpath"
    )
    args = parser.parse_args()

    downsample_and_save(**args)

    """
    usage: python gen_16kHz_ACE_RIR.py -i [ACE RIR dirpath] -o [output dirpath]
    """
