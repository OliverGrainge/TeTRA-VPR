from dataloaders.VPREval import VPREval
from config import DataConfig, ModelConfig, EvalConfig 
from models.helper import get_model
from models.transforms import get_transform
import torch 
import argparse
import pytorch_lightning as pl
from tabulate import tabulate



def _load_model_and_transform(args): 
    if args.preset is not None: 
        model = get_model(preset=args.preset) 
        transform = get_transform(preset=args.preset)

        return model, transform
    else: 
        model = get_model(backbone_arch=args.backbone_arch, agg_arch=args.agg_arch)
        transform = get_transform(augmentation_level="None", image_size=args.image_size)


    if args.weights_path is not None: 
        state_dict = torch.load(args.weights_path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

    for param in model.parameters(): 
        param.requires_grad = False 
    
    if hasattr(model, "freeze"): 
        model.freeze()
    return model, transform



def eval(args): 
    model, transform = _load_model_and_transform(args)
    module = VPREval(
        model=model, 
        transform=transform, 
        val_set_names=args.val_set_names, 
        val_dataset_dir=args.val_dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Initialize a PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="auto", 
        precision="bf16-mixed",
        max_epochs=1,  # Set the number of epochs
        logger=False,  # Disable logging if not needed
        enable_progress_bar=True,
    )
    
    # Use the trainer to validate the module
    trainer.validate(module)

if __name__ == "__main__": 
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser()
    for config in [DataConfig, ModelConfig, EvalConfig]: 
        parser = config.add_argparse_args(parser)
    args = parser.parse_args()

    eval(args)
