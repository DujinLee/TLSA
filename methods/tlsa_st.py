"""TLSA+ST -- self-training on top of the aligned label space.

Stage 2 of the pipeline.  It reads the label space produced by
:mod:`methods.tlsa`, builds the *universal classifier* over
``source classes + predicted target-private classes``, and adapts the frozen
CLIP backbone with AdaptFormer adapters using reliable pseudo-labels:

* the teacher (EMA of the student) labels a clean view of each target image,
* balanced top-k selection keeps the most confident samples *per class*,
* the student is trained on an augmented view with cross-entropy,
* the teacher is EMA-updated once per epoch.
"""

import copy
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import (Normalize, RandomHorizontalFlip,
                                    RandomRotation, RandomVerticalFlip,
                                    ToPILImage)

from datasets import dataset_classes
from methods.alignment import text_information, text_information_blip
from methods.base import BaseMethod
from methods.tlsa import (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD,
                          canonicalize_classnames)
from models import BLIP2_MODELS, BLIP_MODELS, CLIP_MODELS, build_backbone
from models.classifier import ProtoCLS
from templates import get_templates, templates_types
from tools.utils import run_directory


def default_alignment_path(cfg):
    """Alignment result of the matching ``--method TLSA`` run."""
    return os.path.join(run_directory(cfg, method='TLSA'), 'alignment.json')


def imbalance_ratio(class_counts):
    """max/min class frequency; ``inf`` when some selected class is empty."""
    if not class_counts:
        return float('nan')
    counts = np.array(list(class_counts.values()), dtype=float)
    return float(counts.max() / counts.min()) if counts.min() > 0 else float('inf')


def select_topk_by_class(pseudo_labels, confidence, k, mask):
    """Keep the ``k`` most confident samples of every predicted class.

    Classes with fewer than ``k`` samples are kept in full, which is what makes
    the selection balanced rather than dominated by frequent classes.
    """
    for class_label in set(label.item() for label in pseudo_labels):
        indices = [idx for idx, label in enumerate(pseudo_labels)
                   if label.item() == class_label]
        if len(indices) < k:
            mask[indices] = True
            continue
        for i in confidence[indices].topk(k)[1]:
            mask[indices[i]] = True
    return mask


class TLSA_ST(BaseMethod):
    """Stage 2: self-training with the universal classifier."""

    require_source = False
    require_target = True
    require_target_labels = True   # for pseudo-label diagnostics only
    text_templetes = 'ensemble'
    threshold_mode = 'fixed'
    score_mode = 'MLS'

    def __init__(self, cfg, tuning_config=None) -> None:
        cfg.classifier_head = 'prototype'
        assert cfg.backbone in CLIP_MODELS + BLIP_MODELS + BLIP2_MODELS, \
            f'backbone must be a CLIP/BLIP model but got {cfg.backbone}'

        super().__init__(cfg, tuning_config=tuning_config)
        self.cfg = cfg
        self.tuning_config = tuning_config
        self.backbone, self.vis_processors, self.txt_processors = build_backbone(
            cfg.backbone, tuning_config=tuning_config)
        self.backbone = self.backbone.to(self.device)
        if cfg.fixed_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.alignment = self.load_alignment(cfg)
        self.target_dict = self.alignment['target_dict']
        self.classnames = canonicalize_classnames(
            self.get_classnames(cfg) + list(self.alignment['target_private_set']))
        self.templates = self.get_templates(cfg)
        print(f'{len(self.classnames)} universal classes '
              f'({self.num_classes} source + '
              f'{len(self.alignment["target_private_set"])} target-private)')

        self.momentum = cfg.ema_momentum
        self.criterion_target = nn.CrossEntropyLoss(reduction='sum')
        self.classifier = ProtoCLS(self.feature_dim, len(self.classnames))
        weights, self.text_features, self.text_labels = self.encode_classnames(self.classnames)
        self.classifier.fc.weight.data = weights
        self.classifier = self.classifier.to(self.device)

    # ------------------------------------------------------------------ #
    def load_alignment(self, cfg):
        path = cfg.alignment_result or default_alignment_path(cfg)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'alignment result not found at {path}. Run the training-free '
                f'stage first (--method TLSA), or pass --alignment_result.')
        print(f'loading label space alignment from {path}')
        with open(path, 'r') as f:
            return json.load(f)

    def get_classnames(self, cfg):
        return dataset_classes[cfg.dataset](cfg.data_dir, cfg.source_domain,
                                            cfg.target_domain, cfg.n_share,
                                            cfg.n_source_private).classnames

    def get_templates(self, cfg):
        assert self.text_templetes in templates_types
        return get_templates(cfg.dataset, self.text_templetes)

    def encode_classnames(self, classnames):
        if self.cfg.backbone in BLIP_MODELS + BLIP2_MODELS:
            return text_information_blip(self.backbone, self.txt_processors,
                                         classnames, self.templates, self.device)
        return text_information(self.backbone, classnames, self.templates, self.device)

    # ------------------------------------------------------------------ #
    def before_training(self, cfg):
        self.eval()
        if self.fixed_backbone:
            self.backbone.eval()

        self.ema_c = copy.deepcopy(self.classifier).requires_grad_(False)
        self.ema_f = copy.deepcopy(self.backbone).requires_grad_(False)
        self.global_step = 0

    def encode_views(self, target_images):
        """Return ``(student_features, teacher_features)``.

        The student sees an augmented view, the teacher a clean one.
        """
        if self.cfg.backbone in BLIP_MODELS + BLIP2_MODELS:
            to_pil = ToPILImage()
            processed = torch.cat([
                self.vis_processors["eval"](to_pil(img.cpu())).unsqueeze(0).to(self.device)
                for img in target_images])
            samples = {"image": processed}
            student = self.backbone.visual.extract_features(samples, mode="image").image_embeds_proj
            teacher = self.ema_f.visual.extract_features(samples, mode="image").image_embeds_proj
            return student, teacher

        augment = torch.nn.Sequential(
            RandomVerticalFlip(p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=float(np.random.choice([30, 60, 120, 150]))))
        target_images = Normalize(CLIP_PIXEL_MEAN, CLIP_PIXEL_STD)(target_images)
        return self.backbone.visual(augment(target_images)), self.ema_f.visual(target_images)

    def forward(self, batched_inputs):
        self.global_step += 1
        target_images = batched_inputs['target_images'].to(self.device)
        target_labels = batched_inputs['target_labels'].to(self.device)
        batch_size = len(target_images)

        student_features, teacher_features = self.encode_views(target_images)
        teacher_logits = self.ema_c(teacher_features)
        student_logits = self.ema_c(student_features)

        pseudo_labels = teacher_logits.sort(-1)[1][:, -1]
        probabilities = F.softmax(teacher_logits, 1)
        confidence, predictions = torch.max(probabilities, -1)

        # balanced top-k reliable pseudo-label selection
        class_counts = Counter(predictions.tolist())
        mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        mask = select_topk_by_class(pseudo_labels, confidence,
                                    k=int(batch_size / len(class_counts)) + 1,
                                    mask=mask)

        loss_target = self.criterion_target(student_logits[mask], pseudo_labels[mask])

        # the teacher is updated once per epoch over the target domain
        if self.alignment['total'] < (self.global_step * self.cfg.batch_size):
            self.global_step = 0
            self.ema_update()

        return {'loss_target': loss_target,
                **self.pseudo_label_diagnostics(pseudo_labels, target_labels,
                                                mask, class_counts, predictions)}

    @torch.no_grad()
    def pseudo_label_diagnostics(self, pseudo_labels, target_labels, mask,
                                 class_counts, predictions):
        """Logging only -- ground-truth labels never enter the loss."""
        pseudo_labels = pseudo_labels.clone()
        target_labels = target_labels.clone()
        pseudo_labels[pseudo_labels >= self.num_classes] = self.num_classes
        target_labels[target_labels >= self.num_classes] = self.num_classes

        shared = target_labels < self.num_classes
        private = target_labels >= self.num_classes
        accuracy_shared = (pseudo_labels[shared] == target_labels[shared]).sum() / shared.sum()
        accuracy_private = ((pseudo_labels >= self.num_classes) & private).sum() / private.sum()
        hscore = 2 * accuracy_shared * accuracy_private / (accuracy_shared + accuracy_private)

        diagnostics = {'acc_ps': (pseudo_labels[mask] == target_labels[mask]).sum() / mask.sum(),
                       'h-score': hscore}
        if self.cfg.log_imbalance:
            selected = Counter(predictions[mask].tolist())
            diagnostics['imbalance_before'] = torch.tensor(imbalance_ratio(class_counts))
            diagnostics['imbalance_after'] = torch.tensor(imbalance_ratio(selected))
        return diagnostics

    def after_backward(self):
        if self.classifier_type == 'prototype':
            self.classifier.weight_norm()

    @torch.no_grad()
    def ema_update(self):
        for student, teacher in zip(self.classifier.parameters(), self.ema_c.parameters()):
            teacher.data = student.data * (1.0 - self.momentum) + teacher.data * self.momentum
        for student, teacher in zip(self.backbone.parameters(), self.ema_f.parameters()):
            teacher.data = student.data * (1.0 - self.momentum) + teacher.data * self.momentum

    # ------------------------------------------------------------------ #
    def before_predict(self, cfg=None):
        pass

    def predict(self, batched_inputs):
        result_dict = {'predict_labels': None, 'predict_labels_without_ood': None,
                       'features': None, 'logits': None, 'iid_scores': None}
        with torch.no_grad():
            images = batched_inputs['test_images'].to(self.device)
            if self.cfg.backbone in BLIP_MODELS + BLIP2_MODELS:
                to_pil = ToPILImage()
                processed = torch.cat([
                    self.vis_processors["eval"](to_pil(img.cpu())).unsqueeze(0).to(self.device)
                    for img in images])
                features = self.backbone.extract_features({"image": processed},
                                                          mode='image').image_embeds_proj
            else:
                features = self.backbone.visual(
                    Normalize(CLIP_PIXEL_MEAN, CLIP_PIXEL_STD)(images))

            logit = self.classifier(features)
            predict_labels = torch.max(logit, -1)[1]
            predict_labels_without_ood = copy.deepcopy(predict_labels)
            predict_labels[self.predict_ood_indexs(logits=logit)] = self.num_classes

            result_dict.update({'features': features,
                                'logits': logit,
                                'predict_labels': predict_labels,
                                'predict_labels_without_ood': predict_labels_without_ood,
                                'iid_scores': self.get_iid_scores(logit)})
        return result_dict

    def get_ood_scores(self, logits):
        probs = F.softmax(logits, dim=-1).sort(-1)[0]
        return probs[:, -1] - probs[:, -2]

    def predict_confident_indexs(self, logits, num_classes):
        return self.get_ood_scores(logits) > (1 / self.num_classes)

    def get_iid_scores(self, logits):
        max_logits, preds = torch.max(logits, -1)
        max_logits[preds >= self.num_classes] = -max_logits[preds >= self.num_classes]
        return max_logits

    def predict_ood_indexs(self, logits):
        return self.get_iid_scores(logits) < 0
