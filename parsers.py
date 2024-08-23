import argparse
import os
import yaml

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)


import argparse

def model_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Model configuration arguments")

    # Adding arguments for the model configuration
    parser.add_argument('--backbone_arch', type=str, default=config['Model']['backbone_arch'],
                        help='Backbone architecture for the model')
    parser.add_argument('--pretrained', type=bool, default=config['Model']['pretrained'],
                        help='Use pretrained weights')
    parser.add_argument('--layers_to_freeze', type=int, default=config['Model']['layers_to_freeze'],
                        help='Number of layers to freeze')
    parser.add_argument('--layers_to_crop', type=list, default=config['Model']['layers_to_crop'],
                        help='Layers to crop')
    parser.add_argument('--agg_arch', type=str, default=config['Model']['agg_arch'],
                        help='Aggregation architecture')

    # Nested dictionary for agg_config
    parser.add_argument('--agg_config_in_channels', type=int,
                        default=config['Model']['agg_config']['in_channels'],
                        help='Number of input channels for aggregation')
    parser.add_argument('--agg_config_out_channels', type=int,
                        default=config['Model']['agg_config']['out_channels'],
                        help='Number of output channels for aggregation')
    parser.add_argument('--agg_config_s1', type=int, default=config['Model']['agg_config']['s1'],
                        help='Stride 1 for aggregation')
    parser.add_argument('--agg_config_s2', type=int, default=config['Model']['agg_config']['s2'],
                        help='Stride 2 for aggregation')

    return parser

def training_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Training configuration arguments")

    # Adding Training configuration arguments
    parser.add_argument('--accelerator', type=str, default=config['Training']['accelerator'],
                        help='Type of accelerator to use (e.g., gpu, cpu)')
    parser.add_argument('--devices', type=list, default=config['Training']['devices'],
                        help='Devices to use for training')
    parser.add_argument('--precision', type=int, default=config['Training']['precision'],
                        help='Precision for training (e.g., 16 for mixed precision)')
    parser.add_argument('--max_epochs', type=int, default=config['Training']['max_epochs'],
                        help='Maximum number of epochs for training')
    parser.add_argument('--fast_dev_run', type=bool, default=config['Training']['fast_dev_run'],
                        help='Run a quick development run')

    parser.add_argument('--lr', type=float, default=config['Training']['lr'], help='Learning rate')
    parser.add_argument('--optimizer', type=str, default=config['Training']['optimizer'],
                        help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=config['Training']['weight_decay'],
                        help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=config['Training']['momentum'],
                        help='Momentum for optimizer')
    parser.add_argument('--warmup_steps', type=int, default=config['Training']['warmup_steps'],
                        help='Number of warmup steps')

    parser.add_argument('--milestones', type=list, default=config['Training']['milestones'],
                        help='List of epoch milestones for learning rate schedule')
    parser.add_argument('--lr_mult', type=float, default=config['Training']['lr_mult'],
                        help='Learning rate multiplier')
    parser.add_argument('--loss_name', type=str, default=config['Training']['loss_name'],
                        help='Name of the loss function')
    parser.add_argument('--miner_name', type=str, default=config['Training']['miner_name'],
                        help='Name of the miner function')
    parser.add_argument('--miner_margin', type=float, default=config['Training']['miner_margin'],
                        help='Margin for miner function')
    parser.add_argument('--faiss_gpu', type=bool, default=config['Training']['faiss_gpu'],
                        help='Use GPU for FAISS operations')

    parser.add_argument('--monitor', type=str, default=config['Training']['monitor'],
                        help='Metric to monitor during training')

    return parser

def dataloader_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Dataloader configuration arguments")

    # Adding Dataloader configuration arguments
    parser.add_argument('--batch_size', type=int, default=config['Dataloader']['batch_size'],
                        help='Batch size for dataloader')
    parser.add_argument('--img_per_place', type=int, default=config['Dataloader']['img_per_place'],
                        help='Number of images per place')
    parser.add_argument('--min_img_per_place', type=int, default=config['Dataloader']['min_img_per_place'],
                        help='Minimum images per place')
    parser.add_argument('--cities', type=list, default=config['Dataloader']['cities'],
                        help='List of cities')
    parser.add_argument('--image_size', type=list, default=config['Dataloader']['image_size'],
                        help='Size of the images (width, height)')
    parser.add_argument('--num_workers', type=int, default=config['Dataloader']['num_workers'],
                        help='Number of workers for data loading')
    parser.add_argument('--shuffle_all', type=bool, default=config['Dataloader']['shuffle_all'],
                        help='Shuffle all data')
    parser.add_argument('--random_sample_from_each_place', type=bool,
                        default=config['Dataloader']['random_sample_from_each_place'],
                        help='Randomly sample from each place')
    parser.add_argument('--show_data_stats', type=bool, default=config['Dataloader']['show_data_stats'],
                        help='Show data statistics')
    parser.add_argument('--val_set_names', type=list, default=config['Dataloader']['val_set_names'],
                        help='Validation set names')

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model, Training, and Dataloader arguments")

    # Add model arguments
    parser = model_arguments(parser)

    # Add training arguments
    parser = training_arguments(parser)

    # Add dataloader arguments
    parser = dataloader_arguments(parser)

    # Parse the arguments
    args = parser.parse_args()
    print(args)