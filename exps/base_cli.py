# Copyright (c) Megvii Inc. All rights reserved.
import os
from argparse import ArgumentParser
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.ema import EMACallback
from utils.torch_dist import all_gather_object, synchronize

from .base_exp import BEVDepthLightningModel


def run_cli(model_class=BEVDepthLightningModel,
            exp_name='base_exp',
            use_ema=False,
            ckpt_path=None):
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-p',
                               '--predict',
                               dest='predict',
                               action='store_true',
                               help='predict model on testing set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=6,
                        strategy='ddp',
                        # strategy='ddp_find_unused_parameters_false',
                        num_sanity_val_steps=0,
                        check_val_every_n_epoch=100,
                        gradient_clip_val=5,
                        limit_val_batches=0.25,
                        #limit_train_batches=0.001,
                        #limit_val_batches=0.001,
                        log_every_n_steps=1,
                        enable_checkpointing=True,
                        precision=16,
                        default_root_dir=os.path.join('./outputs/', exp_name))
    args = parser.parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = model_class(**vars(args))
    pretrained_model = torch.load(args.ckpt_path)['state_dict']
    model_dict = model.state_dict()
    for k, v in pretrained_model.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v
    model.load_state_dict(model_dict)
    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs)
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback, ModelSummary(max_depth=3)])
    else:
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ModelCheckpoint(save_weights_only=True), ModelSummary(max_depth=3)])
    if args.evaluate:
        # trainer.test(model, ckpt_path=args.ckpt_path)
        trainer.test(model)
    elif args.predict:
        predict_step_outputs = trainer.predict(model, ckpt_path=args.ckpt_path)
        all_pred_results = list()
        all_img_metas = list()
        for predict_step_output in predict_step_outputs:
            for i in range(len(predict_step_output)):
                all_pred_results.append(predict_step_output[i][:3])
                all_img_metas.append(predict_step_output[i][3])
        synchronize()
        len_dataset = len(model.test_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        model.evaluator._format_bbox(all_pred_results, all_img_metas,
                                     os.path.dirname(args.ckpt_path))
    else:
        trainer.fit(model)
        trainer.save_checkpoint("outputs/frustum.ckpt")
