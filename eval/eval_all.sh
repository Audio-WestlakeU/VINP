# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: This is a Bash script speech dereverberation evaluation.

#!/bin/bash


SCRIPT_DIR=$(realpath "$(dirname "$0")")
cd "$SCRIPT_DIR" || exit

test_dir=""
ref_dir=""

while getopts ":r:i:" opt; do
  case $opt in
    r)
      ref_dir="$OPTARG"
      ;;
    i)
      test_dir="$OPTARG"
      ;;
    \?)
      echo "unavalible parameter: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "-$OPTARG needed" >&2
      exit 1
      ;;
  esac
done

echo "test dir: $test_dir"
echo "reference dir: $ref_dir"

python -W ignore eval_DNSMOS_normed.py -t "$test_dir"
echo "DNSMOS DONE"

if [ -z "$ref_dir" ]; then
  exit 1
fi

python -W ignore eval_objective_normSRMR.py -r "$ref_dir" -o "$test_dir"
echo "Obfective Metrics DONE"

# usage: bash eval/eval_all.sh -r [reference dirpath] -i [speech dirpath]