
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
import utils

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from models import helper
import os
import yaml
import argparse
from parsers import model_arguments, training_arguments, dataloader_arguments, quantize_arguments
import sys
from pretrain import VPRModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'NeuroCompress')))
from NeuroPress import QLayers as Q
from NeuroPress import postquantize
torch.set_float32_matmul_precision('medium')


def get_qlayers(args):
    qlinear = getattr(Q, args.qlinear) if args.qlinear else None
    qconv = getattr(Q, args.qconv) if args.qconv else None
    return qlinear, qconv

if __name__ == '__main__':
    # the datamodule contains train and validation dataloaders,
    # refer to ./dataloader/GSVCitiesDataloader.py for details
    # if you want to train on specific cities, you can comment/uncomment
    # cities from the list TRAIN_CITIES
    parser = argparse.ArgumentParser(description="Model, Training, and Dataloader arguments")
    parser = model_arguments(parser)
    parser = training_arguments(parser)
    parser = dataloader_arguments(parser)
    parser = quantize_arguments(parser)
    args = parser.parse_args()

    args.load_checkpoint = "/home/oliver/Documents/github/QuantPlaceFinder/Logs/PreTraining/resnet18/lightning_logs/version_3/checkpoints/resnet18_convap_MultiSimilarityLoss_epoch(15)_step(1872)_R1[0.8436]_R5[0.9447].ckpt"

    # Instantiate the datamodule with parsed arguments
    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        cities=args.cities,
        shuffle_all=args.shuffle_all,
        random_sample_from_each_place=args.random_sample_from_each_place,
        image_size=args.image_size,
        num_workers=args.num_workers,
        show_data_stats=args.show_data_stats,
        val_set_names=args.val_set_names,
    )

    model = VPRModel(
        # ---- Backbone
        backbone_arch=args.backbone_arch,
        pretrained=args.pretrained,
        layers_to_freeze=args.layers_to_freeze,
        layers_to_crop=args.layers_to_crop,

        # ---- Aggregator
        agg_arch=args.agg_arch,  # CosPlace, NetVLAD, GeM, AVG
        agg_config={
            'convap': {
                'in_channels': args.agg_config_in_channels,
                'out_channels': args.agg_config_out_channels,
                's1': args.agg_config_s1,
                's2': args.agg_config_s2,
            },
            'gem': {
                'p': args.agg_config_p,
            },
            'cosplace': {
                'in_dim': args.agg_config_in_channels,
                'out_dim': args.agg_config_out_dim,
            },
                        'fully_connected': {
                'in_channels': args.agg_config_in_channels,
                'spatial_dims': (7, 7),
                'out_dim': 512,
            }
        },

        # ---- Train hyperparameters
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        warmpup_steps=args.warmup_steps,
        milestones=args.milestones,
        lr_mult=args.lr_mult,

        # ----- Loss
        loss_name=args.loss_name,
        miner_name=args.miner_name,
        miner_margin=args.miner_margin,
        faiss_gpu=args.faiss_gpu,
        search_precision=args.search_precision
    )

    state_dict = torch.load(args.load_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict["state_dict"])


    #qlinear, qconv = get_qlayers(args)
    #print(f"Quantizing with {qlinear} and {qconv} weights")
    #postquantize(model.backbone, qlinear=qlinear, qconv=qconv)

    # model params saving using Pytorch Lightning
    # we save the best 3 models according to Recall@1 on pittsburgh val
    checkpoint_cb = ModelCheckpoint(
        monitor=args.monitor,
        filename=f'{model.encoder_arch}' +
                 '_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max',)

    # Instantiate a trainer with parsed arguments
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,  # gpu
        default_root_dir=f'./Logs/PostTraining/{model.encoder_arch}',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs N validation steps before starting training
        precision=args.precision,  # we use half precision to reduce  memory usage (and 2x speed on RTX)
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[checkpoint_cb],  # we run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        fast_dev_run=args.fast_dev_run,  # comment if you want to start training the network and saving checkpoints
    )
    trainer.validate(model=model, datamodule=datamodule)
    # Run the trainer with the model and datamodule
    trainer.fit(model=model, datamodule=datamodule)