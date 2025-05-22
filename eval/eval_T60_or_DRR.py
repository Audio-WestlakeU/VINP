# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: calculate MAE and RMSE of RT60 or DRR estimates.

import json
import numpy as np
import argparse

def main(output_fpath:str,reference_fpath:str):

    with open(output_fpath,'r',encoding='utf-8') as f:
        est = json.load(f)
        
    with open(reference_fpath,'r',encoding='utf-8') as f:
        ref = json.load(f)
    
    
    MAE_list = []
    MSE_list = []
    
    for key in ref.keys():
        MAE_list.append(np.abs(est[key]-ref[key]).item())
        MSE_list.append((np.abs(est[key]-ref[key]).item())**2)
    
    MAE_final = np.mean(MAE_list)
    RMSE_final = np.sqrt(np.mean(MSE_list))
    
    with open(output_fpath+'_eval.txt','w',encoding='utf-8') as f:
        print(f'MAE:{MAE_final},RMSE:{RMSE_final}',file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_fpath",
    )
    parser.add_argument(
        "-r",
        "--reference_fpath",
    )
    args = parser.parse_args()

    main(args.output_fpath,args.reference_fpath)
    
    """
    usage: 
    python eval_T60_or_DRR.py -o [output_json_filepath] -r [reference_json_filepath]
    """