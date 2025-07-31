# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: calculate objective metrics.

from os.path import join
from glob import glob
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import argparse
import numpy as np
import datetime
from pystoi import stoi
from srmrpy import srmr


def mean_std(data):
    mean = np.mean(data)
    std = np.std(data)
    return mean, std


def eval_metrics(a):

    output_root = a.out_path
    ref_root = a.ref_path
    sr = a.sr

    data = {
        "filename": [],
        "wbpesq": [],
        "estoi": [],
        "srmr": [],
    }

    # Evaluate standard metrics
    out_files = sorted(
        glob("{}/*.flac".format(output_root)) + glob("{}/*.wav".format(output_root))
    )

    for out_file in tqdm(out_files, desc="Calculating metrics"):
        filename = out_file.split("/")[-1]
        x, sr_read = read(join(ref_root, filename))

        assert sr == sr_read
        x = x - np.mean(x)
        x = x / np.max(np.abs(x))
        x_method, _ = read(join(output_root, filename))

        if x.shape[-1] > x_method.shape[-1]:
            x_method = np.pad(x_method, (0, x.shape[-1] - x_method.shape[-1]))
        else:
            x = np.pad(x, (0, x_method.shape[-1] - x.shape[-1]))
        x_method = x_method - np.mean(x_method)
        x_method = x_method / np.max(np.abs(x_method))

        data["filename"].append(filename)
        data["wbpesq"].append(pesq(sr, x, x_method, "wb"))
        data["estoi"].append(stoi(x, x_method, sr, extended=True))
        data["srmr"].append(srmr(x_method, sr, norm=True, fast=False)[0])

    df = pd.DataFrame(data)

    # Print results
    with open(join(output_root, f"_ave_metrics_offline_normSRMR.txt"), "w") as f:
        print(datetime.datetime.now(), file=f)
        print(output_root, file=f)
        print(
            "WBPESQ: {:.3f} ± {:.3f}".format(*mean_std(df["wbpesq"].to_numpy())), file=f
        )
        print(
            "ESTOI: {:.3f} ± {:.3f}".format(*mean_std(df["estoi"].to_numpy())), file=f
        )
        print("SRMR: {:.3f} ± {:.3f}".format(*mean_std(df["srmr"].to_numpy())), file=f)


def main():
    print("Calculating the metrics of wavs..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", "-o", required=True)  # output .wav folder
    parser.add_argument("--ref_path", "-r", required=True)  # reference .wav folder
    parser.add_argument("--sr", default=16000)  # sample rate

    args = parser.parse_args()

    eval_metrics(args)


if __name__ == "__main__":
    main()

"""
usage: python eval_objective_normSRMR.py -r [ref_dirpath] -o [output_dirpath]

"""
