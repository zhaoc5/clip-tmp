# MODEL=EVA02-CLIP-B-16
MODEL=EVA02-CLIP-bigE-14-plus
PRETRAINED=eva_clip
LAION_2B_DATA_PATH="/workspace/code/clip-tmp/downloads/train/{00000..00331}.tar"

IMG_EMB_PATH="/workspace/code/clip-tmp/cc-train/clip_cls_emb"
TEXT_EMB_PATH="/workspace/code/clip-tmp/cc-train/clip_cls_emb/text"


CUDA_VISIBLE_DEVICES=1 python training/main.py \
        --val-data=${LAION_2B_DATA_PATH} \
        --val-num-samples=3000000 \
        --batch-size=4 \
        --workers=8 \
        --model=${MODEL} \
        --force-custom-clip \
        --pretrained=${PRETRAINED} \
        --extract-features \
        --img-emb-path=${IMG_EMB_PATH} \
        --text-emb-path=${TEXT_EMB_PATH} \
        --save-interval=250 \
        --enable-deepspeed