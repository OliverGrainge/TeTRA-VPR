from dataloaders.Eval import Eval
from models.helper import get_model
import os
import pytorch_lightning as pl 
import argparse
from omegaconf import OmegaConf
from models.transforms import get_transform
import torch 
from tabulate import tabulate

def _parseargs(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def _load_checkpoint(model, checkpoint_path):
    assert os.path.exists(
        checkpoint_path
    ), f"Backbone weights path {checkpoint_path} does not exist"
    sd = torch.load(checkpoint_path, weights_only=False)["state_dict"]
    model.load_state_dict(sd)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model

def _freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()
    return model

def _get_transform(conf): 
    return get_transform(augmentation_level="None", image_size=conf.image_size)

def _load_model(conf):
    model = get_model(
        backbone_arch=conf.backbone_arch,
        agg_arch=conf.agg_arch,
        image_size=conf.image_size,
    )
    if conf.pretrain_checkpoint:
        os.path.exists(
            conf.pretrain_checkpoint
        ), f"Pretrain checkpoint path {conf.pretrain_checkpoint} does not exist"
        model = _load_checkpoint(model, conf.pretrain_checkpoint)
    model = _freeze_model(model)
    return model


def _setup_eval(conf):
    model = _load_model(conf)
    transform = _get_transform(conf)

    eval_module = Eval(
        model=model, 
        test_set_names=conf.test_set_names, 
        test_dataset_dir=conf.test_dataset_dir, 
        image_size=conf.image_size, 
        num_workers=conf.num_workers, 
        transform=transform,
        batch_size=conf.batch_size, 
        k_values=conf.k_values, 
        matching_precision=conf.matching_precision,
    )

    trainer = pl.Trainer(
        accelerator="auto", 
        precision="bf16-mixed", 
    )

    return trainer, eval_module


if __name__ == "__main__": 
    args = _parseargs()
    conf = OmegaConf.load(args.config)
    trainer, eval_module = _setup_eval(conf)
    trainer.test(eval_module)
            # Print table
    config_basename = os.path.basename(args.config)
    print(f"\nTest Results: {config_basename}")
    print(tabulate(eval_module.results, headers=eval_module.headers, tablefmt="grid"))
