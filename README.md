# VINP [Submitted to IEEE/ACM Trans. on TASLP]

## Introduction

This repo is the official PyTorch implementation of **'VINP: Variational Bayesian Inference with Neural Speech Prior for Joint ASR-Effective Speech Dereverberation and Blind RIR Identification'**, which has been submitted to IEEE/ACM Trans. on TASLP.

Codes will be uploaded later.

[Paper](https://arxiv.org/abs/2502.07205) | [Code](https://github.com/Audio-WestlakeU/VINP) | [DEMO](https://audio.westlake.edu.cn/Research/VINP.htm) 

<!-- ## 2. Usage


### 2.1. Prepare Environment

Please see `requirements.txt`.

### 2.2. Prepare Datasets

#### 2.2.1. Training Set and Validation Set

We build the training set and validation set in the same way. 

1. Prepare reverberant and direct-path RIRs using `dataset/gen_rir.py` as
```
python ./dataset/gen_rir.py --[config_key] [config_val] 
```
where the details is provided in `config/rir.json`

2. Prepare a list of file paths (in `.txt` format) for the source speech (in `.wav` or `.flac` format), simulated RIR pairs (in `.npz` format), and noise (in `.wav` or `.flac` format) using `dataset/gen_fpath_txt.py` as
```
python ./dataset/gen_fpath_txt.py --i [folder path] --o [.txt path] --ext [extension name]
```

#### 2.2.2. Test Set for Dereverberation

Prepare the official single-channel test sets of [REVERB Challenge Dataset](https://reverb2014.audiolabs-erlangen.de/).

#### 2.2.3. Test Set for Blind RIR Identification

1. Prepare the RIRs of the 'Single' subfolder in [ACE Challenge](http://www.ee.ic.ac.uk/naylor/ACEweb/).

2. Generate the test set using `dataset/noisy_dataset_1chl_torch_ACE.py` as
```
```


### 2.3. Training

1. Edit the config file (for example: `config/OSPN.toml` and `config/TCNSAS.toml`).

2. Start training as

```
torchrun --standalone --nnodes=1 --nproc_per_node=[number of GPUs] train.py -c [config file path] -p [save path]
```

3. Resume training

```
torchrun --standalone --nnodes=1 --nproc_per_node=[number of GPUs] train.py -c [config file path] -p [save path] -r
```

### 2.4. Pretrained Checkpoints

```
torchrun --standalone --nnodes=1 --nproc_per_node=[number of GPUs] train.py -c [config file path] -p [save path] --start_ckpt [pretrained model file path]
```

### 2.4. Speech Dereverberation and Blind RIR identification

### 2.5. Evaluation

#### 2.5.1 Speech Quality

1. Download the source codes of [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS).

2. When reference waveforms are available, run
```
sh eval/eval_all.sh [reference dirpath] [output dirpath]
```

Otherwise, run 
```
sh eval/eval_all.sh [output dirpath] [output dirpath]
```

#### 2.5.2 ASR Evaluation

#### 2.5.3 RT60 and DRR Evaluation -->

## Results

### Speech Dereverberation Results on REVERB

<img src="figure/Result_REVERB.png" width="1000">

### Blind RIR Identification Results on SimACE

<img src="figure/Result_SimACE.png" width="520">

<!-- ## 5. References-->

## Citation

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
