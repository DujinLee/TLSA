import os
import time
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from engine.dataloader import build_data_loaders
from engine.evaluator import UniDAEvaluator
from engine.optimizer import build_optimizer
from engine.scheduler import build_lr_scheduler
from methods import method_classes
from tools.utils import run_directory, run_file, save_as_json


class DefaultTrainer(ABC):
    def __init__(self, cfg, tuning_config=None):
        self.model = self.build_model(cfg, tuning_config=tuning_config)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.lr_scheduler = self.build_lr_scheduler(cfg, self.optimizer)

    def build_model(self, cfg, tuning_config=None):
        return method_classes[cfg.method](cfg, tuning_config=tuning_config)

    def build_optimizer(self, cfg, model):
        return build_optimizer(model,
                               cfg.optimizer,
                               cfg.base_lr,
                               cfg.weight_decay,
                               sgd_momentum=cfg.momentum,
                               backbone_multiplier=cfg.backbone_multiplier,
                               clip_norm_value=cfg.clip_norm_value)

    def build_lr_scheduler(self, cfg, optimizer):
        return build_lr_scheduler(optimizer,
                                  cfg.lr_scheduler,
                                  cfg.warmup_iter,
                                  cfg.max_iter,
                                  warmup_type=cfg.warmup_type,
                                  warmup_lr=cfg.warmup_min_lr)

    def _write_metrics(self, metrics):
        print('  '.join(f'{key}: {value}' for key, value in metrics.items()))

    @abstractmethod
    def build_data_loaders(self, cfg):
        return

    @abstractmethod
    def build_evaluator(self, cfg):
        return

    @abstractmethod
    def train(self, cfg):
        return

    @abstractmethod
    def test(self, cfg):
        return

    @abstractmethod
    def load(self, checkpoint_path=None):
        return


class UniDaTrainer(DefaultTrainer):
    def __init__(self, cfg, tuning_config=None):
        super().__init__(cfg, tuning_config=tuning_config)
        (self.source_data_loader, self.target_data_loader, self.test_data_loader,
         self.val_data_loader, self.vlm_data_loader) = self.build_data_loaders(cfg)
        self.evaluator = self.build_evaluator(cfg)
        self.max_iter = cfg.max_iter
        self.cfg = cfg
        self.tuning_config = tuning_config
        self.wandb_run = None
        if self.use_features:
            self.model.backbone = torch.nn.Identity()
            if cfg.ft_last_layer:
                self.model.backbone = self.model.partial_model

    def build_data_loaders(self, cfg):
        feature_dir = os.path.join(cfg.feature_dir,
                                   f'features-imgAug_{cfg.image_augmentation}',
                                   cfg.backbone.replace('/', ''),
                                   cfg.dataset)
        source_feature_path = os.path.join(feature_dir, f'{cfg.source_domain}.pth')
        target_feature_path = os.path.join(feature_dir, f'{cfg.target_domain}.pth')
        if ((cfg.fixed_backbone or cfg.ft_last_layer)
                and os.path.exists(source_feature_path)
                and os.path.exists(target_feature_path)):
            self.use_features = True
            print('Use pre-extracted features as dataloader')
            cfg.num_workers = 0
        else:
            self.use_features = False
            print('Use I/O images as dataloader')
            source_feature_path = None
            target_feature_path = None

        return build_data_loaders(cfg.dataset,
                                  cfg.data_dir,
                                  cfg.source_domain,
                                  cfg.target_domain,
                                  cfg.n_share,
                                  cfg.n_source_private,
                                  cfg.image_augmentation,
                                  cfg.backbone,
                                  cfg.no_balanced,
                                  cfg.batch_size,
                                  cfg.num_workers,
                                  source_feature_path=source_feature_path,
                                  target_feature_path=target_feature_path,
                                  test_feature_path=target_feature_path,
                                  val_feature_path=source_feature_path)

    def build_evaluator(self, cfg):
        return UniDAEvaluator(cfg.n_share + cfg.n_source_private)

    def load(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.model.get_save_checkpoint_dir()
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)

    # ------------------------------------------------------------------ #
    def loader_dict(self, step=0):
        return {'source_data_loader': self.source_data_loader,
                'target_data_loader': self.target_data_loader,
                'test_data_loader': self.test_data_loader,
                'val_data_loader': self.val_data_loader,
                'blip_data_loader': self.vlm_data_loader,
                'step': step}

    def setup_and_test(self):
        """Run ``before_training`` and evaluate, without any gradient step.

        This is the entry point for the training-free stage (TLSA).
        """
        self.model.before_training(cfg=self.loader_dict())
        self.test()

    def init_wandb(self):
        import wandb
        self.wandb_run = wandb.init(
            project=self.cfg.wandb_project,
            config={'method': self.cfg.method,
                    'backbone': self.cfg.backbone,
                    'learning_rate': self.cfg.base_lr,
                    'adapter': self.tuning_config.ffn_adapt,
                    'adapter_bottleneck': self.tuning_config.ffn_num,
                    'dataset': self.cfg.dataset,
                    'source_domain': self.cfg.source_domain,
                    'target_domain': self.cfg.target_domain,
                    'n_share': self.cfg.n_share,
                    'n_source_private': self.cfg.n_source_private,
                    'batch_size': self.cfg.batch_size,
                    'max_iter': self.max_iter,
                    'fixed_backbone': self.cfg.fixed_backbone})
        return wandb

    def train(self, cfg=None):
        start_time = time.time()
        total_parameters = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        adapter = sum(p.numel() for name, p in self.model.named_parameters() if 'adapt' in name)
        print(f'trainable / total parameters: {trainable / total_parameters:.4f}')
        print(f'adapter parameters: {adapter}')
        print(f'trainable parameters: {trainable}')

        self.model.before_training(cfg=self.loader_dict())

        source_loader_iter = iter(self.source_data_loader)
        target_loader_iter = iter(self.target_data_loader)

        wandb = self.init_wandb() if self.cfg.wandb else None
        logged_hscores = []

        for step in tqdm(range(self.max_iter)):
            source_images, source_labels = None, None
            if self.model.require_source:
                try:
                    source_batch = next(source_loader_iter)
                except StopIteration:
                    source_loader_iter = iter(self.source_data_loader)
                    source_batch = next(source_loader_iter)
                source_images, source_labels = source_batch['img'], source_batch['label']

            target_images, target_indexs, target_batch = None, None, None
            if self.model.require_target:
                try:
                    target_batch = next(target_loader_iter)
                except StopIteration:
                    target_loader_iter = iter(self.target_data_loader)
                    target_batch = next(target_loader_iter)
                target_images, target_indexs = target_batch['img'], target_batch['idx']

            batched_inputs = {'source_images': source_images,
                              'source_labels': source_labels,
                              'target_images': target_images,
                              'target_indexs': target_indexs}
            if self.model.require_target_labels and target_batch is not None:
                batched_inputs['target_labels'] = target_batch['label']

            loss_dict = self.model(batched_inputs=batched_inputs)
            if loss_dict is None:
                continue

            if isinstance(loss_dict, torch.Tensor):
                losses, loss_dict = loss_dict, {'total_loss': loss_dict}
            elif 'loss_target' in loss_dict:
                losses = loss_dict['loss_target']
            else:
                losses = sum(loss_dict.values())

            if wandb is not None and 'h-score' in loss_dict:
                logged_hscores.append(loss_dict['h-score'])
                if (step + 1) % self.cfg.wandb_log_every == 0:
                    wandb.log({'loss_target': loss_dict['loss_target'],
                               'h-score': torch.mean(torch.stack(logged_hscores)),
                               'pseudo_label_accuracy': loss_dict['acc_ps']})
                    logged_hscores = []

            self.optimizer.zero_grad()
            losses.backward()

            metrics = {key: value.detach().cpu().item() for key, value in loss_dict.items()}
            metrics['step'] = f'{step}/{self.max_iter}'
            metrics['lr'] = self.lr_scheduler.get_last_lr()[-1]
            self._write_metrics(metrics)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.model.after_backward()

        self.training_time = time.time() - start_time
        self.model.after_training()

    # ------------------------------------------------------------------ #
    def test(self, cfg=None):
        self.model.before_predict(cfg={'val_data_loader': self.val_data_loader,
                                       'test_data_loader': self.test_data_loader})
        self.evaluator.reset()

        logits, iid_scores = [], []
        true_labels, predict_labels, predict_labels_without_ood = [], [], []

        for batch_datas in tqdm(self.test_data_loader, desc='test'):
            result_dict = self.model.predict(batched_inputs={'test_images': batch_datas['img']})
            self.evaluator.process(batch_datas['label'],
                                   result_dict['predict_labels'],
                                   result_dict['predict_labels_without_ood'],
                                   result_dict['iid_scores'],
                                   result_dict['features'])

            true_labels.append(batch_datas['label'].cpu().detach())
            for buffer, key in ((logits, 'logits'),
                                (iid_scores, 'iid_scores'),
                                (predict_labels, 'predict_labels'),
                                (predict_labels_without_ood, 'predict_labels_without_ood')):
                if result_dict[key] is not None:
                    buffer.append(result_dict[key].cpu().detach())

        if not self.cfg.eval_only:
            torch.save({'true_labels': torch.cat(true_labels),
                        'target_logits': torch.cat(logits) if logits else None,
                        'iid_scores': torch.cat(iid_scores) if iid_scores else None,
                        'predict_labels': torch.cat(predict_labels) if predict_labels else None,
                        'predict_labels_without_ood': (torch.cat(predict_labels_without_ood)
                                                       if predict_labels_without_ood else None)},
                       run_file(self.cfg, 'scores.pth'))

        results = self.evaluator.evaluate()
        self._write_metrics(results)
        save_as_json(results, run_file(self.cfg, 'results.json'))
        save_as_json(vars(self.cfg), run_file(self.cfg, 'config.json'))
        print(f'results saved to {run_directory(self.cfg)}')
        return results
