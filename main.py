
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
from parsers import model_arguments, training_arguments, dataloader_arguments

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """

    def __init__(self,
                #---- Backbone
                backbone_arch=config['Model']['backbone_arch'],
                pretrained=config['Model']['pretrained'],
                layers_to_freeze=config['Model']['layers_to_freeze'],
                layers_to_crop=config['Model']['layers_to_crop'],
                
                #---- Aggregator
                agg_arch=config['Model']['agg_arch'], #CosPlace, NetVLAD, GeM, AVG
                agg_config=config['Model']['agg_config'],
                
                #---- Train hyperparameters
                lr=config['Training']['lr'],
                optimizer=config['Training']['optimizer'],
                weight_decay=config['Training']['weight_decay'],
                momentum=config['Training']['momentum'],
                warmpup_steps=config['Training']['warmup_steps'],
                milestones=config['Training']['milestones'],
                lr_mult=config['Training']['lr_mult'],
                
                #----- Loss
                loss_name=config['Training']['loss_name'],
                miner_name=config['Training']['miner_name'],
                miner_margin=config['Training']['miner_margin'],
                faiss_gpu=config['Training']['faiss_gpu'],
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay, 
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
        return [optimizer], [scheduler]
    
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}
    
    # This is called at the end of eatch training epoch
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    def on_validation_epoch_start(self):
        # Initialize or reset the list to store validation outputs
        self.validation_outputs = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        # store the outputs
        self.validation_outputs.append(descriptors.detach().cpu())
        return descriptors.detach().cpu()
    
    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs."""
        dm = self.trainer.datamodule

        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list
        if len(dm.val_datasets) == 1:
            val_step_outputs = [self.validation_outputs]
        else:
            val_step_outputs = self.validation_outputs

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)

            num_references = val_dataset.num_references
            num_queries = val_dataset.num_queries
            ground_truth = val_dataset.ground_truth

            # split to ref and queries    
            r_list = feats[:num_references]
            q_list = feats[num_references:]

            recalls_dict, predictions = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )

            self.log(f'{val_set_name}/R1', recalls_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', recalls_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', recalls_dict[10], prog_bar=False, logger=True)

            del r_list, q_list, feats, num_references, ground_truth

        # Clear the outputs after processing
        self.validation_outputs.clear()
        print('\n\n')
            
            
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
            'in_channels': args.agg_config_in_channels,
            'out_channels': args.agg_config_out_channels,
            's1': args.agg_config_s1,
            's2': args.agg_config_s2,
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
    )

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
        default_root_dir=f'./LOGS/{model.encoder_arch}',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs N validation steps before starting training
        precision=args.precision,  # we use half precision to reduce  memory usage (and 2x speed on RTX)
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[checkpoint_cb],  # we run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        fast_dev_run=args.fast_dev_run,  # comment if you want to start training the network and saving checkpoints
    )

    # Run the trainer with the model and datamodule
    trainer.fit(model=model, datamodule=datamodule)