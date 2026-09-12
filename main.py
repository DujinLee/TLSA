"""Entry point for TLSA.

Stage 1 (training-free label space alignment):

    python main.py --method TLSA --dataset office31 \
        --source_domain amazon --target_domain dslr --n_share 10 --n_source_private 10

Stage 2 (self-training on the aligned label space):

    python main.py --method TLSA_ST --dataset office31 \
        --source_domain amazon --target_domain dslr --n_share 10 --n_source_private 10 \
        --ffn_adapt --max_iter 2500 --batch_size 64 --base_lr 0.01
"""

import os
import sys

# The bundled ``clip`` package (AdaptFormer-enabled) must win over any copy
# installed in site-packages, so the repository root goes first on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faiss  # noqa: F401  must be imported before torch (used by the evaluator)
import torch
from easydict import EasyDict

from configs import parser
from engine.trainer import UniDaTrainer
from methods import method_classes
from tools.utils import set_random_seed

torch.set_num_threads(1)


def build_tuning_config(args):
    """AdaptFormer configuration (defaults follow the AdaptFormer paper)."""
    return EasyDict(
        ffn_adapt=args.ffn_adapt,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args.ffn_num,
        d_model=768 if args.backbone == "ViT-B/16" else 1024,
        ffn_start=args.ffn_start,
        ffn_end=args.ffn_end,
    )


def main(args):
    set_random_seed(args.seed)
    trainer = UniDaTrainer(args, tuning_config=build_tuning_config(args))

    if args.eval_only:
        trainer.load(args.checkpoint)
        trainer.test()
        return

    if method_classes[args.method].training_free:
        # no gradient step: run the alignment pass, then evaluate
        trainer.setup_and_test()
        return

    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main(parser.parse_args())
