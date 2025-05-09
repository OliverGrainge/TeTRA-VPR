
BACKBONE_WEIGHT_PATH="checkpoints/TeTRA-pretrain/Student[VitbaseT322]-Teacher[DinoV2]-Aug[Severe]/epoch=17-step=136250-train_loss=0.0460-qfactor=1.00.ckpt"
#python finetune.py --agg_arch boq --backbone_arch vitbaset --image_size 322 322 --batch_size 80 --num_workers 12 --freeze_backbone True --backbone_checkpoint $BACKBONE_WEIGHT_PATH --quant_schedule linear --max_epochs 10 --pbar True
#python finetune.py --agg_arch boq --backbone_arch vitbaset --image_size 322 322 --batch_size 80 --num_workers 12 --freeze_backbone True --backbone_checkpoint $BACKBONE_WEIGHT_PATH --quant_schedule cosine --max_epochs 10  --pbar True
#python finetune.py --agg_arch boq --backbone_arch vitbaset --image_size 322 322 --batch_size 80 --num_workers 12 --freeze_backbone True --backbone_checkpoint $BACKBONE_WEIGHT_PATH --quant_schedule none --max_epochs 10  --pbar True

#python finetune.py --agg_arch boq --backbone_arch dinot --image_size 322 322 --batch_size 80 --num_workers 12 --freeze_backbone True  --quant_schedule logistic --max_epochs 10 --pbar True
#python finetune.py --agg_arch boq --backbone_arch dinoboqt --image_size 322 322 --batch_size 80 --num_workers 12 --freeze_backbone True  --quant_schedule logistic --max_epochs 10 --pbar True


python finetune.py --agg_arch boq --backbone_arch dinot --image_size 322 322 --batch_size 64 --num_workers 12 --quant_schedule logistic --max_epochs 40 --pbar True
python finetune.py --agg_arch boq --backbone_arch dinoboqt --image_size 322 322 --batch_size 64 --num_workers 12 --quant_schedule logistic --max_epochs 40 --pbar True