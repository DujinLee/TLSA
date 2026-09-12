#!/usr/bin/env bash
# Stage 2: self-training on the aligned label space (TLSA+ST).
#
# Requires the matching `--method TLSA` run to have finished, since the label
# space is read from that run's alignment.json.
#
#   bash scripts/run_tlsa_st.sh office31         # one benchmark
#   bash scripts/run_tlsa_st.sh                  # all four benchmarks
set -euo pipefail

BACKBONE="${BACKBONE:-ViT-B/16}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BASE_LR="${BASE_LR:-0.01}"
ADAPTER_DIM="${ADAPTER_DIM:-64}"
SEED="${SEED:-1}"

# "<n_share> <n_source_private> <max_iter> <scenario>" -- Tab. 2 and Tab. 3
office31_splits=("10 10 2500 open-partial" "10 0 2500 open"
                 "31 0 5000 closed" "10 21 5000 partial")
officehome_splits=("10 5 2500 open-partial" "15 0 2500 open"
                   "65 0 5000 closed" "25 40 5000 partial")
visda_splits=("6 3 5000 open-partial" "6 0 5000 open"
              "12 0 10000 closed" "6 6 10000 partial")
domainnet_splits=("150 50 5000 open-partial" "150 0 5000 open"
                  "345 0 10000 closed" "150 195 10000 partial")

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
            read -r n_share n_source_private max_iter scenario <<< "${split}"
            echo "=== ${dataset} ${source_domain} -> ${target_domain} (${scenario}) ==="
            python main.py \
                --method TLSA_ST \
                --dataset "${dataset}" \
                --source_domain "${source_domain}" \
                --target_domain "${target_domain}" \
                --n_share "${n_share}" \
                --n_source_private "${n_source_private}" \
                --backbone "${BACKBONE}" \
                --batch_size "${BATCH_SIZE}" \
                --base_lr "${BASE_LR}" \
                --max_iter "${max_iter}" \
                --ffn_adapt \
                --ffn_num "${ADAPTER_DIM}" \
                --seed "${SEED}" \
                --save_checkpoint
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
