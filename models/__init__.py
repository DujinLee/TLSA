import os
import sys

# The bundled `clip` package (with AdaptFormer adapters) must shadow any copy
# installed in site-packages, so the repository root goes first on sys.path.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import clip

from models.resnet import ResBase
from models.classifier import CLS, ProtoCLS, Projection, ProtoNormCLS
from models.vit import vit_base, vit_base_dino, deit_base
from lavis.models import load_model_and_preprocess


CLIP_MODELS = clip.available_models()
DINOv2_MODELS = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']
BLIP_MODELS = ["blip_base", "blip_large"]
BLIP2_MODELS = ["blip2_giga", "blip2_large"]
backbone_names = ['resnet18', 'resnet50'] + CLIP_MODELS + DINOv2_MODELS + ['vit_base', 'vit_base_dino', 'deit_base'] + BLIP_MODELS + BLIP2_MODELS
head_names = ['linear', 'mlp', 'prototype', 'protonorm']


def build_backbone(name, tuning_config=None):
    """
    build the backbone for feature exatraction
    """
    vis_processors, txt_processors = None, None
    if 'resnet' in name:
        return ResBase(option=name), vis_processors, txt_processors
    elif name in CLIP_MODELS:
        model, _ = clip.load(name, tuning_config=tuning_config)
    elif name in DINOv2_MODELS:
        import torch
        model = torch.hub.load('facebookresearch/dinov2', name)
        # model = torch.hub.load('/data1/deng.bin/coding/JUSTforLearning/dinov2', name, source='local')
    elif name == 'vit_base':
        model = vit_base(pretrained=False)
    elif name == 'vit_base_dino':
        model = vit_base_dino()
    elif name == 'deit_base':
        model = deit_base()
    elif name in BLIP_MODELS:
        model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type=name, is_eval=True)
    elif name in BLIP2_MODELS:
        model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type=name, is_eval=True)
    else:
        raise RuntimeError(f"Model {name} not found; available models = {backbone_names}")
    
    return model.float(), vis_processors, txt_processors


def build_head(name, in_dim, out_dim, hidden_dim=2048, temp=0.05):
    if name == 'linear':
        return CLS(in_dim, out_dim)
    elif name == 'mlp':
        return Projection(in_dim, feat_dim=out_dim, hidden_mlp=hidden_dim)
    elif name == 'prototype':
        return ProtoCLS(in_dim, out_dim, temp=temp)
    elif name == 'protonorm':
        return ProtoNormCLS(in_dim, out_dim, temp=temp)
    else:
        raise RuntimeError(f"Model {name} not found; available models = {head_names}")