# VINP [Submitted to IEEE/ACM Trans. on TASLP]

## Introduction

This repo is the official PyTorch implementation of **'VINP: Variational Bayesian Inference with Neural Speech Prior for Joint ASR-Effective Speech Dereverberation and Blind RIR Identification'**, which has been submitted to IEEE/ACM Trans. on TASLP.

[Paper](https://arxiv.org/abs/2502.07205) | [Code](https://github.com/Audio-WestlakeU/VINP) | [DEMO](https://audio.westlake.edu.cn/Research/VINP.htm) 




## Before Training

#### Requirements

Please see `requirements.txt`.

#### Prepare Training Set and Validation Set

Step1. Prepare clean source speech and noise recordings in .wav or .flac format.

Step2. Prepare reverberant and direct-path RIRs
```
python dataset/gen_rir.py -c [config/config_gen_rir.json]
```

Step3. Save the list of filepath for the source speech, simulated RIR (.npz), and noise to .txt file
```
python datset/gen_fpath_txt.py -i [dirpath] -o [.txt filepath] -e [filename extension]
```

#### Prepare Test Set for Dereverberation

Prepare the official single-channel test sets of [REVERB Challenge Dataset](https://reverb2014.audiolabs-erlangen.de/).

#### Prepare Test Set for Blind RIR Identification

Step1. Prepare the RIRs of the 'Single' subfolder in [ACE Challenge](http://www.ee.ic.ac.uk/naylor/ACEweb/).

Step2. Downsample the RIRs to 16kHz
```
python datset/gen_16kHz_ACE_RIR.py -i [ACE 'Single' dirpath] -o [saved dirpath]
```

Step3. Save the list of filepath for the source speech, ACE RIR, and noise to .txt file
```
python datset/gen_fpath_txt.py -i [dirpath] -o [.txt filepath] -e [filename extension]
```

Step4. Generate the test set (consists of reverberant speech and labels)
```
python dataset/gen_SimACE_testset.py --[keyword] [arg]
```


## Training (Codes for training will be uploaded later)

Step1. Edit the config file (for example: `config/config_VINP_oSpatialNet.toml` and `config/config_VINP_TCNSAS.toml`).

Step2. Run

```
# train from scratch
torchrun --standalone --nnodes=1 --nproc_per_node=[number of GPUs] train.py -c [config filepath] -p [saved dirpath]

# resume training
torchrun --standalone --nnodes=1 --nproc_per_node=[number of GPUs] train.py -c [config filepath] -p [saved dirpath] -r 

# use pretrained checkpoints
torchrun --standalone --nnodes=1 --nproc_per_node=[number of GPUs] train.py -c [config filepath] -p [saved dirpath] --start_ckpt [pretrained model filepath]
```


## Inference

Run
```
python enhance_rir_avg.py -c [config filepath] --ckpt [list of checkpoints] -i [reverberant speech dirpath] -o [output dirpath] -d [GPU id]
```


## Evaluation

Evaluation results are saved to the output folder.

#### Speech Quality

For SimData, run
```
bash eval/eval_all.sh -i [speech dirpath] -r [reference dirpath]
```

For RealData, the reference is not available. Run 
```
bash eval/eval_all.sh -i [speech dirpath]
```


#### ASR

For SimData, run
```
python eval/eval_ASR_REVERB_SimData.py -i [speech dirpath] -m [whisper model name (tiny small medium)]
```

For RealData, run
```
python eval/eval_ASR_REVERB_RealData.py -i [speech dirpath] -m [whisper model name (tiny small medium)]
```

#### RT60 and DRR

Step1. Estimate RT60 and DRR using
```
python estimate_T60_DRR.py -i [estimated RIR dirpath]
```

Step2. Run
```
python eval/eval_T60_or_DRR.py -o [estimated RT60 or DRR .json] -r [reference RT60 or DRR .json]
```

## Results

### Speech Dereverberation Results on REVERB

<img src="figure/Result_REVERB.png" width="1000">
<img src="figure/Curve_REVERB.png" width="1000">

### Blind RIR Identification Results on SimACE

<img src="figure/Result_SimACE.png" width="500">

<img src="figure/Curve_SimACE.png" width="400">

## Citations

If you find our work helpful, please cite
```
@misc{wang2025vinpvariationalbayesianinference,
      title={VINP: Variational Bayesian Inference with Neural Speech Prior for Joint ASR-Effective Speech Dereverberation and Blind RIR Identification}, 
      author={Pengyu Wang and Ying Fang and Xiaofei Li},
      year={2025},
      eprint={2502.07205},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2502.07205}, 
}
```