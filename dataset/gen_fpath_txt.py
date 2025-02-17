# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: code for save filepath list to .txt.

from glob import glob
import os
from jsonargparse import ArgumentParser


def save_addresses_to_txt(addresses: list, out_path: str):
    with open(out_path, "w") as file:
        for address in addresses:
            file.write(address + "\n")

    return None


def gen_txt(dir_path: str, filetype: str):

    target_path = sorted(
        glob(os.path.join(dir_path, "**", "*{}".format(filetype)), recursive=True)
    )

    return target_path


def main(input_dir, output_path, ext):

    addresses = gen_txt(input_dir, ext)
    save_addresses_to_txt(addresses, output_path)

    return None


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input_dir", required=True, type=str, help="input dirpath"
    )
    parser.add_argument(
        "-o", "--output_path", required=True, type=str, help="output filepath"
    )
    parser.add_argument(
        "-e", "--ext", required=True, type=str, help="filename extension"
    )
    args = parser.parse_args()
    main(**args)

    """
    python gen_fpath_txt.py -i [dirpath] -o [saved .txt filepath] -e [extension]
    """
