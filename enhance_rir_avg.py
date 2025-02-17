# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: speech dereverberation and blind RIR identification.

from glob import glob
from tqdm import tqdm
import os
import torch
from acoustics.feature import (
    load_wav,
    save_wav,
    norm_amplitude,
)
from collections import defaultdict
import toml
from typing import Dict
from jsonargparse import ArgumentParser
import json
from trainer_inferencer.utils import initialize_module
from pathlib import Path


def save_config_to_file(args, file_path):
    with open(file_path, "w") as json_file:
        json.dump(args.__dict__, json_file, indent=4)


def average_checkpoints(checkpoints):
    param_sums = defaultdict(lambda: 0)
    num_checkpoints = len(checkpoints)
    for ckpt in checkpoints:
        if "use_ema" in ckpt and ckpt["use_ema"]:
            state_dict = ckpt["model_ema"]
        else:
            state_dict = ckpt["model"]

        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            param_sums[new_key] += value.float()

    averaged_state_dict = {}
    for key, sum_value in param_sums.items():
        averaged_state_dict[key] = sum_value / num_checkpoints

    return averaged_state_dict


@torch.no_grad()
def enhance_avg(
    input_path: str,
    output_path: str,
    model: Dict,
    EM_algo: Dict,
    acoustic: Dict,
    ckpt: list,
    device: str,
    *args,
    **kwargs
):

    fpath_input = sorted(
        glob("{}/**/*{}".format(input_path, ".flac"), recursive=True)
    ) + sorted(glob("{}/**/*{}".format(input_path, ".wav"), recursive=True))
    basename_input = [os.path.basename(i) for i in fpath_input]
    N_seq = len(basename_input)

    TF = initialize_module(acoustic["path"], acoustic["args"])
    sr = TF.sr

    mymodel = initialize_module(model["path"], model["args"])
    mymodel.to(device)

    checkpoints = [torch.load(ckpt_path, map_location="cpu") for ckpt_path in ckpt]
    averaged_state_dict = average_checkpoints(checkpoints)
    mymodel.load_state_dict(averaged_state_dict, strict=True)

    mymodel.eval()

    rkem = initialize_module(EM_algo["path"], EM_algo["args"])
    total_steps = 0
    total_likelihood = 0
    count = 0
    for fpath_input_n in tqdm(fpath_input):
        count += 1
        basename_input_n = os.path.basename(fpath_input_n)
        fpath_out_n_normed = os.path.join(output_path, "normed", basename_input_n)
        fpath_out_rir = os.path.join(output_path, "rir", basename_input_n)

        input_wav = load_wav(fpath_input_n, sr)
        input_wav, scale = norm_amplitude(input_wav)

        steps = 0
        likelihood = 0

        spch_est, rir, steps, likelihood = rkem.process(input_wav, mymodel, TF, device)

        total_steps += steps
        total_likelihood += likelihood

        spch_est *= scale

        save_wav(spch_est / spch_est.abs().max(), fpath_out_n_normed, sr)
        save_wav(rir / rir.abs().max(), fpath_out_rir, sr)

    with open(
        os.path.join(os.path.join(output_path, "_metainfo.txt")), "w", encoding="utf-8"
    ) as f:
        print("Average Steps:", total_steps / N_seq, file=f)
        print("Average Likelihood:", total_likelihood / N_seq, file=f)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Configuration .toml file"
    )
    parser.add_argument("--ckpt", required=True, type=list, help="checkpoint path")
    parser.add_argument(
        "-i", "--input_path", required=True, type=str, help="input path"
    )
    parser.add_argument(
        "-o", "--output_path", required=True, type=str, help="output path"
    )
    parser.add_argument("-d", "--device", required=False, type=str, default="cuda:0")

    args = parser.parse_args()

    config_path = Path(args.config).expanduser().absolute()
    os.makedirs(args.output_path, exist_ok=True)
    config = toml.load(config_path.as_posix())
    with open(os.path.join(args.output_path, "config.toml"), "w") as f:
        toml.dump(config, f)

    save_config_to_file(args, os.path.join(args.output_path, "config.json"))

    enhance_avg(**args, **config)

    """
    usage:
    python enhance_rir_avg.py -c [config filepath] --ckpt [checkpoint filepath list] -i [reverberant speech dirpath] -o [output dirpath] -d [device ID]
    """
