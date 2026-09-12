import argparse
import os

from configs import default
from datasets import dataset_classes
from methods import method_classes
from methods.alignment import FREQUENCY_STATISTICS, FUSION_STRATEGIES
from models import backbone_names, head_names

parser = argparse.ArgumentParser()

###########################
# Directory Config (modify if using your own paths)
###########################
parser.add_argument(
    "--data_dir",
    type=str,
    default=default.DATA_DIR,
    help="where the dataset is saved",
)
parser.add_argument(
    "--feature_dir",
    type=str,
    default=default.FEATURE_DIR,
    help="where to save pre-extracted features",
)
parser.add_argument(
    "--result_dir",
    type=str,
    default=default.RESULT_DIR,
    help="where to save experiment results",
)
# parser.add_argument(
#     "--config_dir",
#     type=str,
#     default=default.CONFIG_DIR,
#     help="where to load additional paras of a specific method which names of 'method.yaml'. \
#         Note that the paras in method.yaml are not overwritten with the common args.",
# )

###########################
# Method Config
###########################
parser.add_argument(
    "--method",
    type=str,
    default="TLSA",
    choices=method_classes.keys(),
    help="which method to run",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="seed number",
)

###########################
# model Config (methods)
###########################

parser.add_argument(
    "--backbone",
    type=str,
    # default="ViT-L/14@336px",
    default="resnet50",
    choices=backbone_names,
    help="specify the encoder-backbone to use",
)
parser.add_argument(
    "--classifier_head",
    type=str,
    default="prototype",
    choices=head_names,
    help="classifier head architecture",
)
parser.add_argument(
    "--fixed_backbone",
    action="store_true",
    help="wheather fixed backbone during training",
)
parser.add_argument(
    "--fixed_BN",
    action="store_true",
    help="wheather fixed batch normalization layers during training",
)
parser.add_argument(
    "--ft_norm_only",
    action="store_true",
    help="wheather only to finetune normalization layers during training",
)
parser.add_argument(
    "--ft_last_layer",
    action="store_true",
    help="wheather only to finetune the last layer during training",
)
parser.add_argument(
    "--save_checkpoint",
    action="store_true",
    help="wheather fixed batch normalization layers during training",
)
parser.add_argument(
    "--eval_only",
    action="store_true",
    help="wheather evaluation only from loading the default checkpoints",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="path to load checkpoint for eval_only"
)
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="which cuda to be used",
)

###########################
# Training Config (trainer.py)
###########################
# Dataset
parser.add_argument(
    "--dataset",
    type=str,
    default="visda",
    choices=dataset_classes.keys(),
    help="number of train shot",
)
parser.add_argument(
    "--source_domain",
    type=str,
    default="syn",
    help="source domain name",
)
parser.add_argument(
    "--target_domain",
    type=str,
    default="real",
    help="target domain name",
)
parser.add_argument(
    "--n_share",
    type=int,
    default=6,
    help="the number of common classes between source and target domains",
)
parser.add_argument(
    "--n_source_private",
    type=int,
    default=3,
    help="the number of private classes of source domain",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="batch size for test (feature extraction and evaluation)",
)
parser.add_argument(
    "--image_augmentation",
    type=str,
    default='none',
    choices=['none', # only a single center crop
             'flip', # add random flip view
             'randomcrop', # add random crop view
             ],
    help="specify the image augmentation to use.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="number of workers for dataloader",
)
parser.add_argument(
    "--no_balanced",
    action="store_true",
    help="trains with no balanced sampling",
)

# Optimizer
parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    choices=["adamw", "sgd"],
    help="optimizer"
)
parser.add_argument(
    '--base_lr', 
    type=float, 
    default=1e-2,
    help='base learning rate'
)
parser.add_argument(
    '--backbone_multiplier', 
    type=float, 
    default=0.1,
    help='backbone learning rate scaling factor'
)
parser.add_argument(
    '--weight_decay', 
    type=float, 
    default=5e-4,
    help='weight decay'
)
parser.add_argument(
    '--momentum',
    type=float, 
    default=0.9,
    help='sgd momentum'
)
parser.add_argument(
    '--clip_norm_value', 
    type=float, 
    default=0,
    help='clip gradient value'
)
parser.add_argument(
    "--max_iter",
    type=int,
    default=10000,
    help="max iters to run",
)

# lr scheduler
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="cosine",
    choices=["cosine", "linear", "constant"],
    help="optimizer type"
)
parser.add_argument(
    "--warmup_iter",
    type=int,
    default=50,
    help="warmup iter",
)
parser.add_argument(
    "--warmup_type",
    type=str,
    default="linear",
    choices=["linear", "constant"],
    help="warmup type"
)
parser.add_argument(
    '--warmup_min_lr', 
    type=float, 
    default=1e-5,
    help='warmup min lr'
)

parser.add_argument(
    '--ffn_start',  
    default=0, 
    type=int, 
    help='start layer of AdaptFormer'
)
parser.add_argument(
    '--ffn_end',  
    default=12, 
    type=int, 
    help='end layer of AdaptFormer'
)
parser.add_argument(
    '--ffn_adapt',  
    default=False, 
    action='store_true', 
    help='whether activate AdaptFormer'
)
parser.add_argument(
    '--ffn_num',
    default=64, 
    type=int, 
    help='bottleneck middle dimension'
)
###########################
# TLSA Config
###########################
parser.add_argument(
    "--vlm",
    type=str,
    default="blip",
    choices=["blip", "qwen"],
    help="generative VLM used to discover labels on target images. "
         "'blip' is BLIP-VQA through LAVIS (used for the main results); "
         "'qwen' talks to an OpenAI-compatible server (e.g. vLLM serving Qwen3-VL)",
)
parser.add_argument(
    "--vllm_api_url",
    type=str,
    default=os.getenv("VLLM_API_URL", "http://localhost:8000/v1"),
    help="base URL of the OpenAI-compatible server (--vlm qwen)",
)
parser.add_argument(
    "--vllm_model_name",
    type=str,
    default=os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct"),
    help="model name/path served by the endpoint (--vlm qwen)",
)
parser.add_argument(
    "--vlm_max_concurrency",
    type=int,
    default=32,
    help="maximum number of in-flight requests to the VLM server (--vlm qwen)",
)
parser.add_argument(
    "--num_prompts",
    type=int,
    default=5,
    help="size of the question ensemble used for label discovery; "
         "the discovered label is the majority vote over the answers",
)
parser.add_argument(
    "--prompt_set",
    type=str,
    default="default",
    choices=["default", "alt"],
    help="which question set to draw the prompt ensemble from; 'alt' is the "
         "disjoint rephrased set used for the prompt-robustness study",
)
parser.add_argument(
    "--topk",
    type=int,
    default=5,
    help="number of top-ranked candidates considered by semantic label alignment",
)
parser.add_argument(
    "--fusion",
    type=str,
    default="min",
    choices=list(FUSION_STRATEGIES),
    help="how the gap and average thresholds are combined into the prediction "
         "set: 'min' keeps candidates above either threshold (default), 'max' "
         "above both, 'gap'/'avg' use a single criterion",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.01,
    help="scaling factor of the frequency threshold, tau_freq = epsilon * stat(F)",
)
parser.add_argument(
    "--freq_stat",
    type=str,
    default="max",
    choices=list(FREQUENCY_STATISTICS),
    help="reference statistic of the frequency bank used for tau_freq",
)
parser.add_argument(
    "--no_wordnet",
    dest="use_wordnet",
    action="store_false",
    help="skip the WordNet synonym label alignment step",
)

###########################
# Self-training Config (TLSA_ST)
###########################
parser.add_argument(
    "--alignment_result",
    type=str,
    default=None,
    help="path to the alignment.json produced by the TLSA stage; "
         "by default the matching TLSA run directory is used",
)
parser.add_argument(
    "--ema_momentum",
    type=float,
    default=0.98,
    help="momentum of the EMA teacher update",
)
parser.add_argument(
    "--log_imbalance",
    action="store_true",
    help="log the pseudo-label imbalance ratio before/after balanced top-k selection",
)

###########################
# Logging Config
###########################
parser.add_argument(
    "--wandb",
    action="store_true",
    help="log training curves to Weights & Biases",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="tlsa",
    help="wandb project name",
)
parser.add_argument(
    "--wandb_log_every",
    type=int,
    default=10,
    help="how often (in iterations) to log to wandb",
)
