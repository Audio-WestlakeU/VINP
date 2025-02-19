#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: wangpengyu@westlake.edu.cn
### 
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

cd eval/ASR_whisper/

whisper_tag=${2}    # whisper model tag, e.g., small, medium, large, etc
cleaner=whisper_en
hyp_cleaner=whisper_en
nj=1
test_sets="et_simu_1ch"
# decode_options is used in Whisper model's transcribe method
decode_options="{language: en, task: transcribe, temperature: 0, beam_size: 10, fp16: False}"

for x in ${test_sets}; do
    wavscp=./examples/reverb/data/${x}/wav.scp    # path to wav.scp
    outdir=${1}/whisper-${whisper_tag}_outputs/${x}  # path to save output
    gt_text=./examples/reverb/data/${x}/text      # path to groundtruth text file (for scoring only)

    utils/evaluate_asr.sh \
        --whisper_tag ${whisper_tag} \
        --nj ${nj} \
        --gpu_inference true \
        --inference_nj 8 \
        --stage 2 \
        --stop_stage 3 \
        --cleaner ${cleaner} \
        --hyp_cleaner ${hyp_cleaner} \
        --decode_options "${decode_options}" \
        --gt_text ${gt_text} \
        ${wavscp} \
        ${outdir}
done