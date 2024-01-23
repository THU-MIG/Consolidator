#!/usr/bin/env bash         \

# We follow NOAH to construct the training framework. The training result is slightly unstable across multiple experiments or on different machines.

set -x

SRUN_ARGS=${SRUN_ARGS:-""}

#mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \


CKPT="ckpts/vit-b-300ep.pth.tar"
DATASETS=(cifar100 caltech101 dtd oxford_flowers102 svhn sun397 oxford_pet patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele)
LRS=("0.001" "0.002" "0.001" "0.002" "0.01" "0.001" "0.001" "0.002" "0.002" "0.002" "0.001" "0.002" "0.001" "0.002" "0.001" "0.01" "0.01" "0.01" "0.001")
WEIGHT_DECAYS=("0.1" "0.1" "0.1" "0.1" "0" "0" "0.1" "0.0001" "0" "0.1" "0.1" "0.001" "0.1" "0" "0.0001" "0.001" "0.0001" "0.0001" "0")
DRS=("0.0" "0.0" "0.0" "0.8" "0.5" "0.0" "0.8" "0.2" "0.8" "0.8" "0.8" "0.2" "0.8" "0.8" "0.5" "0.2" "0.8" "0.2" "0.5")
GROUP="384"

if [ ! -d ./saves ]
then
  mkdir ./saves
fi

for((i=0;i<${#DATASETS[@]};i++))
do
  WEIGHT_DECAY=${WEIGHT_DECAYS[${i}]}
  LR=${LRS[${i}]}
  DR=${DRS[${i}]}
  DATASET=${DATASETS[${i}]}
  CONFIG=experiments/Consolidator/ViT-B_mocov3_prompt_consolidator_${GROUP}.yaml
  TARGET_DIR=./saves/mocov3_vit-b/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}_consolidator_g${GROUP}_d${DR}
  if [ ! -d ${TARGET_DIR} ]
  then
    mkdir ${TARGET_DIR}
  else
    echo "Dir already exists. Skip ${TARGET_DIR}"
    continue
  fi
  python train.py --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET} --cfg=${CONFIG} --resume=${CKPT} --output_dir=${TARGET_DIR} --batch-size=64 --lr=${LR} --epochs=100 --is_consolidator --weight-decay=${WEIGHT_DECAY} --consolidator_drop_ratio=${DR} --no_aug --mixup=0 --cutmix=0 --direct_resize --smoothing=0 --launcher=none\
  2>&1 | tee -a ${TARGET_DIR}/output.log
done