
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
import yaml
torch.set_float32_matmul_precision('medium')

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)


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
        backbone_config=config["Model"]["backbone_config"],

        # ---- Aggregator
        agg_arch=args.agg_arch,  # CosPlace, NetVLAD, GeM, AVG
        agg_config=config["Model"]["agg_config"],

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
    #state_dict = torch.load("/home/oliver/Documents/github/QuantPlaceFinder/Logs/PreTraining/resnet18/lightning_logs/version_0/checkpoints/resnet18_convap_MultiSimilarityLoss_epoch(17)_step(2106)_R1[0.8525]_R5[0.9491].ckpt", map_location="cpu")
    model.load_state_dict(state_dict["state_dict"])


    # model params saving using Pytorch Lightning
    # we save the best 3 models according to Recall@1 on pittsburgh val
    checkpoint_cb = ModelCheckpoint(
        monitor=args.monitor,
        filename=f'{model.backbone_arch}' +
                 '_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max',)

    # Instantiate a trainer with parsed arguments
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,  # gpu
        default_root_dir=f'./Logs/PostTraining/{model.backbone_arch}',  # Tensorflow can be used to viz
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

    qlinear, qconv = get_qlayers(args)
    print(f"Quantizing with {qlinear} and {qconv} weights")


    qlayer_map = {}
    for name, layer in model.named_modules():
        #print(name)
        if "fc1" in name: 
            if "11" in name or "10" in name or "9" in name or "8" in name:
                qlayer_map[layer] = qlinear
            #print("================================================================================= Quantizing")
        if "fc2" in name: 
            if "11" in name or "10" in name or "9" in name or "8" in name:
                qlayer_map[layer] = qlinear
            #old_weights = layer.weight.data.detach().cpu().numpy().flatten()
            #print("================================================================================= Quantizing")


    #postquantize(model, layer_map=qlayer_map)
    #from optimum.quanto import quantize, qint8
    #quantize(model, weights=qint8, activations=qint8)

    #for name, layer in model.named_modules():
    #    print(name, layer)
    #for name, layer in model.named_modules():
        #if "backbone.model.blocks.1.mlp.fc1" in name: 
            #new_weights = layer.quantize().detach().cpu().numpy().flatten()

    
    #import matplotlib.pyplot as plt 
    #plt.hist(new_weights, bins=254, label="Quantized", alpha=0.7, density=True)
    #plt.hist(old_weights, bins=254, label="old weights", alpha=0.7, density=True)
    #import numpy as np
    #print(len(np.unique(new_weights)), "MAX", np.abs(new_weights).max())
    #plt.legend()
    #plt.show()
    trainer.validate(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)