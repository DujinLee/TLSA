#!/usr/bin/env bash
# Stage 1: training-free label space alignment.
#
#   bash scripts/run_tlsa.sh office31            # one benchmark
#   bash scripts/run_tlsa.sh                     # all four benchmarks
#
# Set VLM=qwen to run the robustness variant against an OpenAI-compatible
# server (see the README) instead of BLIP-VQA.
set -euo pipefail

BACKBONE="${BACKBONE:-ViT-B/16}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-1}"
VLM="${VLM:-blip}"

# "<n_share> <n_source_private> <scenario>" -- the split settings of Tab. 2
office31_splits=("10 10 open-partial" "10 0 open" "31 0 closed" "10 21 partial")
officehome_splits=("10 5 open-partial" "15 0 open" "65 0 closed" "25 40 partial")
visda_splits=("6 3 open-partial" "6 0 open" "12 0 closed" "6 6 partial")
domainnet_splits=("150 50 open-partial" "150 0 open" "345 0 closed" "150 195 partial")

# "<source_domain> <target_domain>"
office31_pairs=("amazon dslr" "amazon webcam" "dslr amazon" "dslr webcam"
                "webcam amazon" "webcam dslr")
officehome_pairs=("Art Clipart" "Art Product" "Art RealWorld"
                  "Clipart Art" "Clipart Product" "Clipart RealWorld"
                  "Product Art" "Product Clipart" "Product RealWorld"
                  "RealWorld Art" "RealWorld Clipart" "RealWorld Product")
visda_pairs=("syn real")
domainnet_pairs=("painting real" "painting sketch" "real painting"
                 "real sketch" "sketch painting" "sketch real")

run_dataset () {
    local dataset=$1
    local -n pairs="${dataset}_pairs"
    local -n splits="${dataset}_splits"
    for pair in "${pairs[@]}"; do
        read -r source_domain target_domain <<< "${pair}"
        for split in "${splits[@]}"; do
            read -r n_share n_source_private scenario <<< "${split}"
            echo "=== ${dataset} ${source_domain} -> ${target_domain} (${scenario}) ==="
            python main.py \
                --method TLSA \
                --dataset "${dataset}" \
                --source_domain "${source_domain}" \
                --target_domain "${target_domain}" \
                --n_share "${n_share}" \
                --n_source_private "${n_source_private}" \
                --backbone "${BACKBONE}" \
                --batch_size "${BATCH_SIZE}" \
                --seed "${SEED}" \
                --vlm "${VLM}"
        done
    done
}

datasets=("$@")
if [[ ${#datasets[@]} -eq 0 ]]; then
    datasets=(office31 officehome visda domainnet)
fi
for dataset in "${datasets[@]}"; do
    run_dataset "${dataset}"
done
