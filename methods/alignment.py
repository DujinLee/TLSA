"""Building blocks of training-free label space alignment.

Step 1  synonym label alignment      -- :func:`synonym_alignment`, :func:`merge_synonym_candidates`
Step 2  semantic label alignment     -- :func:`prediction_set_mask`
Step 3  frequency-based filtering    -- :func:`frequency_threshold`

Plus the CLIP / BLIP text-embedding helpers used to turn a list of class names
into prototype classifier weights.
"""

import clip
import numpy as np
import torch
from nltk.corpus import wordnet as wn
from tqdm.auto import tqdm

FUSION_STRATEGIES = ('min', 'max', 'gap', 'avg')
FREQUENCY_STATISTICS = ('max', 'mean', 'median')


# --------------------------------------------------------------------------- #
# text embeddings
# --------------------------------------------------------------------------- #
@torch.no_grad()
def text_information(model, classnames, templates, device=0):
    """Prompt-ensembled CLIP text embeddings for ``classnames``.

    Returns ``(zeroshot_weights, per_template_embeddings, template_labels)``.
    """
    zeroshot_weights, embeddings, labels = [], [], []
    for label, classname in enumerate(classnames):
        texts = clip.tokenize([t.format(classname) for t in templates]).to(
            next(model.parameters()).device)
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
        embeddings.append(class_embeddings)
        labels += [label] * len(class_embeddings)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(
        next(model.parameters()).device)
    return zeroshot_weights.T, torch.cat(embeddings), torch.tensor(labels)


@torch.no_grad()
def text_information_blip(blip_model, txt_processors, classnames, templates, device=0):
    """BLIP counterpart of :func:`text_information`."""
    zeroshot_weights, embeddings, labels = [], [], []
    for label, classname in enumerate(classnames):
        processed = [txt_processors['eval'](t.format(classname)) for t in templates]
        class_embeddings = blip_model.extract_features(
            {"text_input": processed}, mode='text').text_embeds_proj.to(device)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
        embeddings.append(class_embeddings)
        labels += [label] * len(class_embeddings)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.T, torch.cat(embeddings), torch.tensor(labels)


# --------------------------------------------------------------------------- #
# Step 1: synonym label alignment (WordNet)
# --------------------------------------------------------------------------- #
def get_synsets(phrase):
    """WordNet noun synsets for a (possibly multi-word) phrase."""
    if not phrase:
        return None
    words = phrase.split()
    synsets = wn.synsets(phrase, pos=wn.NOUN)
    if not synsets and len(words) >= 2:
        synsets = wn.synsets(phrase.replace(" ", "_"), pos=wn.NOUN)
        if not synsets:
            synsets = wn.synsets(phrase.replace(" ", "-"), pos=wn.NOUN)
            if not synsets:
                synsets.extend(wn.synsets(words[0], pos=wn.NOUN))
        synsets.extend(wn.synsets(words[-1], pos=wn.NOUN))
    return synsets


def synonym_alignment(candidates, class_synsets):
    """Drop discovered labels that are WordNet synonyms of a source class.

    Returns ``(kept_candidates, synonym_to_source)`` where ``synonym_to_source``
    maps every removed label to the source class it collapses onto.
    """
    remove_list, synonym_to_source = [], {}
    for candidate in tqdm(candidates, desc='synonym alignment', leave=False):
        if candidate is None:
            continue
        candidate_synsets = get_synsets(candidate)
        if not candidate_synsets:
            continue
        for cls, cls_synsets in class_synsets.items():
            if not cls_synsets:
                continue
            if any(syn in cls_synsets for syn in candidate_synsets):
                remove_list.append(candidate)
                synonym_to_source[candidate] = cls
    remove_list = list(set(remove_list)) + [None]
    return ([c for c in set(candidates) if c not in remove_list],
            synonym_to_source)


def merge_synonym_candidates(candidates, same_dict):
    """Collapse candidates sharing identical synsets onto the shortest surface form.

    ``same_dict`` is carried across batches so that the mapping stays consistent
    for the whole target domain.
    """
    candidates = [c for c in candidates if c is not None]
    remove_list = []
    synsets = {cap: get_synsets(cap) for cap in candidates}
    for i, synset_i in synsets.items():
        for j, synset_j in synsets.items():
            if i == j or not synset_i or not synset_j:
                continue
            if synset_i == synset_j and len(i) > len(j):
                remove_list.append(i)
                same_dict[i] = j
    remove_list = set(remove_list)
    return [c for c in set(candidates) if c not in remove_list], same_dict


# --------------------------------------------------------------------------- #
# Step 2: semantic label alignment (instance-adaptive thresholding)
# --------------------------------------------------------------------------- #
def prediction_set_mask(topk_score, fusion='min'):
    """Mask selecting the prediction set ``C`` among the top-k candidates.

    ``topk_score`` is ``[B, k]`` sorted in *ascending* order, so column ``k-1``
    holds the highest similarity.  Two instance-adaptive thresholds are used:

    * ``tau_gap`` -- the score right above the largest drop in the ranking,
    * ``tau_avg`` -- the mean of the top-k scores.

    ``fusion`` selects how they are combined:

    ``min``  keep candidates above *either* threshold (the paper's default,
             equivalent to thresholding at ``min(tau_gap, tau_avg)``),
    ``max``  keep candidates above *both*,
    ``gap``  ``tau_gap`` only,
    ``avg``  ``tau_avg`` only.
    """
    assert fusion in FUSION_STRATEGIES, f'unknown fusion strategy {fusion}'
    k = topk_score.shape[1]

    # tau_gap: position of the largest score drop within the ranking
    gap_index = torch.max(topk_score[:, -(k - 1):] - topk_score[:, -k:-1], 1)[1]
    positions = torch.arange(k, device=topk_score.device)
    positions = positions.unsqueeze(0).expand(gap_index.size(0), -1)
    mask_gap = (positions - gap_index.unsqueeze(1)) > 0

    # tau_avg: local average of the top-k scores
    mask_avg = topk_score - torch.mean(topk_score, dim=1, keepdim=True) > 0

    if fusion == 'min':
        return mask_avg | mask_gap
    if fusion == 'max':
        return mask_avg & mask_gap
    if fusion == 'gap':
        return mask_gap
    return mask_avg


# --------------------------------------------------------------------------- #
# Step 3: frequency-based noisy candidate filtering
# --------------------------------------------------------------------------- #
def frequency_threshold(frequency_bank, statistic='max', epsilon=0.01):
    """``tau_freq = epsilon * stat(F)`` over the frequency bank ``F``."""
    assert statistic in FREQUENCY_STATISTICS, f'unknown statistic {statistic}'
    if not frequency_bank:
        return 0.0
    values = np.array(list(frequency_bank.values()), dtype=float)
    base = {'max': values.max, 'mean': values.mean,
            'median': lambda: float(np.median(values))}[statistic]()
    return epsilon * float(base)
