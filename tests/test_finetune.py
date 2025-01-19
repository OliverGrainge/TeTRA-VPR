import argparse
import os
import sys

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.helper import get_model


def test_finetune():
    from config import DataConfig, ModelConfig, TeTRAConfig
    from dataloaders.TeTRA import TeTRA
    from models.helper import get_model

    torch.set_float32_matmul_precision("medium")

    dataconfig = DataConfig()
    modelconfig = ModelConfig()
    tetraconfig = TeTRAConfig()

    model = get_model(
        tetraconfig.image_size, modelconfig.backbone_arch, modelconfig.agg_arch
    )

    model_module = TeTRA(
        model,
        train_dataset_dir=dataconfig.train_dataset_dir,
        val_dataset_dir=dataconfig.val_dataset_dir,
        batch_size=tetraconfig.batch_size,
        image_size=tetraconfig.image_size,
        num_workers=tetraconfig.num_workers,
        cities=tetraconfig.cities,
        lr=tetraconfig.lr,
        scheduler_type="sigmoid",
    )

    pl.Trainer(
        enable_progress_bar=tetraconfig.pbar,
        strategy="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        precision=tetraconfig.precision,
        max_epochs=tetraconfig.max_epochs,
        reload_dataloaders_every_n_epochs=1,
        fast_dev_run=True,
    )
