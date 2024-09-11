
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
import utils
import time
torch.set_float32_matmul_precision('medium')


from dataloaders.GSVCities import GSVCities
from models import helper
import os
import yaml
import argparse
from parsers import model_arguments, training_arguments, dataloader_arguments

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)


IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

TRAIN_CITIES = [
    'Bangkok', 'BuenosAires', 'LosAngeles', 'MexicoCity', 'OSL', 
    'Rome', 'Barcelona', 'Chicago', 'Madrid', 'Miami', 'Phoenix', 
    'TRT', 'Boston', 'Lisbon', 'Medellin', 'Minneapolis', 'PRG', 
    'WashingtonDC', 'Brussels', 'London', 'Melbourne', 'Osaka', 'PRS'
]

SMALL_TRAIN_CITIES = [
    "London",
    "Melbourne",
    "Boston"
]
            
if __name__ == '__main__':
    # the datamodule contains train and validation dataloaders,
    # refer to ./dataloader/GSVCitiesDataloader.py for details
    # if you want to train on specific cities, you can comment/uncomment
    # cities from the list TRAIN_CITIES
    parser = argparse.ArgumentParser(description="Model, Training, and Dataloader arguments")
    parser = model_arguments(parser)
    parser = training_arguments(parser)
    parser = dataloader_arguments(parser)
    args = parser.parse_args()

    model = torch.nn.Sequential(
        helper.get_backbone(args.backbone_arch, config["Model"]["backbone_config"]),
        helper.get_aggregator(args.agg_arch, config["Model"]["agg_config"]),
        
    )

    model = GSVCities(
        config, 
        model,
        batch_size=args.batch_size,
        img_per_place=args.img_per_place, 
        min_img_per_place=args.min_img_per_place, 
        shuffle_all=args.shuffle_all, 
        image_size=args.image_size, 
        num_workers=args.num_workers, 
        cities=SMALL_TRAIN_CITIES,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=["pitts30k_val"])

    # model params saving using Pytorch Lightning
    # we save the best 3 models according to Recall@1 on pittsburgh val
    checkpoint_cb = ModelCheckpoint(
        monitor=args.monitor,
        filename=f'{args.backbone_arch}' + f'_{args.agg_arch}' + f'_{args.loss_name}' +
                 '_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        mode='max',)

    # Instantiate a trainer with parsed arguments
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,  # gpu
        default_root_dir=f'./Logs/PreTraining/{model.backbone_arch}',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs N validation steps before starting training
        precision='bf16',  # we use half precision to reduce  memory usage (and 2x speed on RTX)
        max_epochs=args.max_epochs,
        #check_val_every_n_epoch=2,  # run validation every epoch
        callbacks=[checkpoint_cb],  # we run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        fast_dev_run=args.fast_dev_run,  # comment if you want to start training the network and saving checkpoint
  )   

    # Run the trainer with the model and datamodule
    trainer.fit(model)