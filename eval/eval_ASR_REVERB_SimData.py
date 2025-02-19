# Author: fangying@westlake.edu.cn
# LastEditors: wangpengyu@westlake.edu.cn

import os
import time
import argparse

def main(input_dir:str,scp_file:str,model:str):

    with open(scp_file, 'r') as f:
        lines = f.readlines()

    f = open(scp_file, 'w')
    for line in lines:
        orgname = os.path.dirname(line.split(' ')[-1])
        line = line.replace(orgname, input_dir)
        f.write(line)
    f.close()

    time.sleep(1)
    os.system(f"eval/ASR_whisper/examples/reverb/evaluate_via_whisper_SimData.sh {input_dir} {model}")
    time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
    )
    parser.add_argument(
        "-m",
        "--model",
    )
    parser.add_argument(
        "-r",
        "--scp_file",
        default='eval/ASR_whisper/examples/reverb/data/et_simu_1ch/wav.scp'
    )
    args = parser.parse_args()

    main(args.input_dir,args.scp_file,args.model)