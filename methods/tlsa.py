"""TLSA -- Training-free Label Space Alignment.

A single pass over the unlabeled target domain that

1. discovers a label for every target image with a generative VLM,
2. removes lexical synonyms of source classes with WordNet (Step 1),
3. removes semantically ambiguous labels with instance-adaptive thresholding
   in CLIP's joint embedding space (Step 2),
4. filters the remaining candidates by occurrence frequency (Step 3),

and finally builds the *universal classifier* over
``source classes + predicted target-private classes``.

No gradient step is taken here; the alignment result is written as
``alignment.json`` in the run directory so that the self-training stage
(:mod:`methods.tlsa_st`) can pick it up.
"""

import copy
import json
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize, ToPILImage
from tqdm.auto import tqdm

from datasets import dataset_classes
from methods.alignment import (frequency_threshold, get_synsets,
                               merge_synonym_candidates, prediction_set_mask,
                               synonym_alignment, text_information,
                               text_information_blip)
from methods.base import BaseMethod
from methods.vlm import build_label_generator
from models import BLIP2_MODELS, BLIP_MODELS, CLIP_MODELS, build_backbone
from models.classifier import ProtoCLS
from templates import get_templates, templates_types
from tools.utils import run_file

CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]

# A few benchmark class names are spelled in a way CLIP's text encoder and
# WordNet handle poorly; we canonicalise them once, for every dataset.
CLASSNAME_ALIASES = {
    'laptop computer': 'laptop',
    'desktop computer': 'desktop',
    'paper notebook': 'notebook',
    'bike helmet': 'helmet',
    'desk lamp': 'lamp',
    'desk chair': 'chair',
    'mug': 'mug cup',
    'mobile phone': 'mobile_phone',
    'back pack': 'backpack',
    'phone': 'telephone',
}


def canonicalize_classnames(classnames):
    return [CLASSNAME_ALIASES.get(name, name) for name in classnames]


class TLSA(BaseMethod):
    """Stage 1: training-free label space alignment."""

    require_source = False
    require_target = True
    training_free = True
    text_templetes = 'ensemble'
    threshold_mode = 'fixed'
    score_mode = 'MLS'

    def __init__(self, cfg, tuning_config=None) -> None:
        cfg.fixed_backbone = True
        cfg.classifier_head = 'prototype'
        assert cfg.backbone in CLIP_MODELS + BLIP_MODELS + BLIP2_MODELS, \
            f'backbone must be a CLIP/BLIP model but got {cfg.backbone}'

        super().__init__(cfg, tuning_config=tuning_config)
        self.backbone, self.vis_processors, self.txt_processors = build_backbone(cfg.backbone)
        self.backbone = self.backbone.to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.cfg = cfg
        self.classnames = canonicalize_classnames(self.get_classnames(cfg))
        self.templates = self.get_templates(cfg)
        print(f'{len(self.classnames)} source classes, {len(self.templates)} templates')

        weights, self.text_features, self.text_labels = self.encode_classnames(self.classnames)
        self.classifier.fc.weight.data = weights
        self.classifier = self.classifier.to(self.device)

    # ------------------------------------------------------------------ #
    # setup helpers
    # ------------------------------------------------------------------ #
    def get_classnames(self, cfg):
        return dataset_classes[cfg.dataset](cfg.data_dir, cfg.source_domain,
                                            cfg.target_domain, cfg.n_share,
                                            cfg.n_source_private).classnames

    def get_templates(self, cfg):
        assert self.text_templetes in templates_types
        return get_templates(cfg.dataset, self.text_templetes)

    def encode_classnames(self, classnames):
        """Prompt-ensembled text embeddings with the configured backbone."""
        if self.cfg.backbone in BLIP_MODELS + BLIP2_MODELS:
            return text_information_blip(self.backbone, self.txt_processors,
                                         classnames, self.templates, self.device)
        return text_information(self.backbone, classnames, self.templates, self.device)

    def encode_images(self, raw_images):
        """L2-normalised image embeddings in the joint embedding space."""
        if self.cfg.backbone in BLIP_MODELS + BLIP2_MODELS:
            to_pil = ToPILImage()
            processed = torch.cat([
                self.vis_processors["eval"](to_pil(img.cpu())).unsqueeze(0).to(self.device)
                for img in raw_images])
            features = self.backbone.extract_features({"image": processed}, mode="image")
            embedding = features.image_embeds_proj
        else:
            embedding = self.backbone.visual(
                Normalize(CLIP_PIXEL_MEAN, CLIP_PIXEL_STD)(raw_images))
        return F.normalize(embedding)

    def get_questions(self):
        """The prompt ensemble: the first ``--num_prompts`` questions."""
        template = 'question' if self.cfg.prompt_set == 'default' else 'question_alt'
        questions = get_templates(None, template)
        return questions[:min(self.cfg.num_prompts, len(questions))]

    # ------------------------------------------------------------------ #
    # Step 1-3: the alignment pass
    # ------------------------------------------------------------------ #
    def before_training(self, cfg):
        self.eval()
        data_loader = cfg["blip_data_loader"]
        loader_iter = iter(data_loader)

        label_generator = build_label_generator(self.cfg, self.device)
        questions = self.get_questions()
        to_pil = ToPILImage()

        # state carried across batches
        source_private_set = set(self.classnames)   # source classes never matched
        shared_dict = {}                            # frequency bank, shared part
        target_private_dict = {}                    # frequency bank, private part
        same_dict = {}                              # candidate -> canonical candidate
        caption_bank = {}                           # target index -> discovered label
        class_synsets = {cls: get_synsets(cls) for cls in self.classnames}
        caption_feature_bank = {cls: self.classifier.fc.weight.data[i]
                                for i, cls in enumerate(self.classnames)}
        sample_is_shared, sample_is_target_private = [], []

        with torch.no_grad():
            for _ in tqdm(range(len(data_loader)), desc='label space alignment'):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(data_loader)
                    batch = next(loader_iter)

                raw_images = batch['img'].to(self.device)
                labels, target_indexs = batch['label'], batch['idx']
                sample_is_target_private.extend(labels >= self.num_classes)
                sample_is_shared.extend(labels < self.num_classes)

                # -- discover labels with the generative VLM -------------- #
                pil_images = [to_pil(img.cpu()) for img in raw_images]
                answer_matrix = np.array(label_generator.generate(pil_images, questions)).T
                discovered = [majority_vote(row) for row in answer_matrix]
                for i, index in enumerate(target_indexs):
                    caption_bank[index.item()] = discovered[i]

                visual_embedding = self.encode_images(raw_images)

                # -- Step 1: synonym label alignment ---------------------- #
                if self.cfg.use_wordnet:
                    candidates, synonym_to_source = synonym_alignment(discovered, class_synsets)
                    source_private_set -= set(synonym_to_source.values())
                else:
                    candidates, synonym_to_source = list(set(discovered)), {}
                candidates, same_dict = merge_synonym_candidates(candidates, same_dict)

                # -- Step 2: semantic label alignment --------------------- #
                batch_candidates = list(set(candidates))
                detector = self.build_candidate_detector(batch_candidates, caption_feature_bank)
                logits = detector(visual_embedding)

                k = min(self.cfg.topk, logits.shape[1])
                sorted_logits = logits.sort(-1)
                topk_score, topk_idx = sorted_logits[0][:, -k:], sorted_logits[1][:, -k:]
                mask = prediction_set_mask(topk_score, fusion=self.cfg.fusion)

                label_space = self.classnames + batch_candidates
                for i in range(visual_embedding.shape[0]):
                    label = discovered[i]
                    if label is None:
                        continue
                    label = same_dict.get(label, label)
                    label = synonym_to_source.get(label, label)

                    # the discovered label is literally a source class
                    if label in self.classnames:
                        shared_dict[label] = shared_dict.get(label, 0) + 1
                        continue

                    prediction_set = topk_idx[i][mask[i]]
                    if torch.sum(prediction_set < self.num_classes) == 0:
                        # no source class in C -> target-private sample
                        if label in batch_candidates:
                            target_private_dict[label] = target_private_dict.get(label, 0) + 1
                    else:
                        # a source class is in C -> the discovered label is shared
                        ranks = torch.arange(k - 1, -1, -1, device=self.device)
                        source_rank = ranks[topk_idx[i] < self.num_classes][-1]
                        matched = label_space[topk_idx[i, -(source_rank + 1)]]
                        source_private_set -= {matched}
                        shared_dict[matched] = shared_dict.get(matched, 0) + 1

        # -- Step 3: frequency-based noisy candidate filtering ------------ #
        frequency_bank = shared_dict | target_private_dict
        for label in frequency_bank:
            canonical = same_dict.get(label)
            if canonical in frequency_bank:
                frequency_bank[canonical] += frequency_bank[label]
                frequency_bank[label] = 0
        tau_freq = frequency_threshold(frequency_bank,
                                       statistic=self.cfg.freq_stat,
                                       epsilon=self.cfg.epsilon)
        target_private_set = [label for label in target_private_dict
                              if frequency_bank[label] > tau_freq]

        print('source_private_set :', source_private_set)
        print('target_private_set :', target_private_set)

        self.save_alignment_result({
            'shared': [t.tolist() for t in sample_is_shared],
            'target_private': [t.tolist() for t in sample_is_target_private],
            'target_private_set': target_private_set,
            'source_private_set': list(source_private_set),
            'total': len(data_loader.dataset),
            'caption_bank': caption_bank,
            'target_dict': frequency_bank,
        })

        # -- build the universal classifier ------------------------------ #
        universal_classnames = self.classnames + list(target_private_set)
        self.classifier = ProtoCLS(visual_embedding.shape[1], len(universal_classnames))
        self.classifier.fc.weight.data = self.encode_classnames(universal_classnames)[0]
        self.classifier = self.classifier.to(self.device)

    def build_candidate_detector(self, batch_candidates, caption_feature_bank):
        """Prototype classifier over ``source classes + this batch's candidates``."""
        detector = ProtoCLS(self.feature_dim, self.num_classes + len(batch_candidates))
        for idx, candidate in enumerate(batch_candidates):
            if candidate not in caption_feature_bank:
                caption_feature_bank[candidate] = self.encode_classnames([candidate])[0]
            detector.fc.weight.data[self.num_classes + idx] = caption_feature_bank[candidate]
        detector.fc.weight.data[:self.num_classes] = self.classifier.fc.weight.data
        return detector.to(self.device)

    def save_alignment_result(self, result):
        path = run_file(self.cfg, 'alignment.json')
        with open(path, 'w') as f:
            json.dump(result, f)
        print(f'alignment result saved to {path}')

    # ------------------------------------------------------------------ #
    # inference -- training-free, so forward() is a no-op
    # ------------------------------------------------------------------ #
    def forward(self, batched_inputs):
        return None

    def before_predict(self, cfg=None):
        pass

    def predict(self, batched_inputs):
        result_dict = {'predict_labels': None, 'predict_labels_without_ood': None,
                       'features': None, 'logits': None, 'iid_scores': None}
        with torch.no_grad():
            features = self.encode_images_for_predict(batched_inputs['test_images'].to(self.device))
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

    def encode_images_for_predict(self, images):
        """Unnormalised features (the prototype head normalises internally)."""
        if self.cfg.backbone in BLIP_MODELS + BLIP2_MODELS:
            to_pil = ToPILImage()
            processed = torch.cat([
                self.vis_processors["eval"](to_pil(img.cpu())).unsqueeze(0).to(self.device)
                for img in images])
            return self.backbone.extract_features({"image": processed},
                                                  mode='image').image_embeds_proj
        return self.backbone.visual(Normalize(CLIP_PIXEL_MEAN, CLIP_PIXEL_STD)(images))

    def get_ood_scores(self, logits):
        probs = F.softmax(logits, dim=-1)
        return (probs[:, self.num_classes:].sort(-1)[0][:, -1]
                - probs[:, :self.num_classes].sort(-1)[0][:, -1])

    def predict_confident_indexs(self, logits, num_classes):
        return self.get_ood_scores(logits) > (1 / self.num_classes)

    def get_iid_scores(self, logits):
        max_logits, preds = torch.max(logits, -1)
        max_logits[preds > self.num_classes] = -max_logits[preds > self.num_classes]
        return max_logits

    def predict_ood_indexs(self, logits):
        return self.get_iid_scores(logits) < 0


def majority_vote(answers):
    """Most frequent non-``None`` answer across the prompt ensemble."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]
