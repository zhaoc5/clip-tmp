# MODEL=EVA02-CLIP-B-16
MODEL=EVA02-CLIP-bigE-14-plus
PRETRAINED=eva_clip
LAION_2B_DATA_PATH="/workspace/code/EVA/data_download/val/{00000..00001}.tar"

IMG_EMB_PATH="/workspace/code/EVA/_output_val-e14p/img"
TEXT_EMB_PATH="/workspace/code/EVA/_output_val-e14p/text"


CUDA_VISIBLE_DEVICES=1 python training/main.py \
        --val-data=${LAION_2B_DATA_PATH} \
        --val-num-samples=12813 \
        --batch-size=4 \
        --workers=8 \
        --model=${MODEL} \
        --force-custom-clip \
        --pretrained=${PRETRAINED} \
        --extract-features \
        --img-emb-path=${IMG_EMB_PATH} \
        --text-emb-path=${TEXT_EMB_PATH} \
        --save-interval=12813 \
        --enable-deepspeed