#!/bin/bash

# Define variables
batch_size=64
num_workers=12
val_set_names=("pitts30k" "amstertime" "tokyo247" "eynsham")

#define the presets to evaluate
presets=("CosPlaces" "DinoSalad" "EigenPlaces" "DinoV2_BoQ" "ResNet50_BoQ")


# define the TeTRA models to evaluate
backbone_archs=("vit_base_PLRBitLinear" "vit_base_PLRBitLinear" "vit_small_PLRBitLinear" "vit_small_PLRBitLinear")
agg_archs=("salad" "gem" "boq" "mixvpr")
image_size=("224" "322")




echo "================================================================================="
echo "=============================== Baseline models ================================"
echo "================================================================================="
for preset in "${presets[@]}"; do
    for val_set in "${val_set_names[@]}"; do
        echo "========================================"
        echo "Running evaluation for preset:"
        echo "  PRESET: $preset"
        echo "  VAL SET: $val_set"
        echo "========================================"
        python eval.py --preset "$preset" --batch_size $batch_size --num_workers $num_workers --val_set_names $val_set --silent True
    done
done


echo " "
echo " "
echo " "
echo " "
echo " "
echo " "
echo " "
echo " "
echo " "
echo " "
echo " "
echo " "

echo "================================================================================="
echo "=============================== TeTRA models ==================================="
echo "================================================================================="
for backbone in "${backbone_archs[@]}"; do
    for agg_arch in "${agg_archs[@]}"; do
        for image_size in "${image_size[@]}"; do
            for val_set in "${val_set_names[@]}"; do
                echo "========================================"
                echo "Running evaluation for model:"
                echo "  BACKBONE_ARCH: $backbone"
                echo "  AGG_ARCH: $agg_arch"
                echo "  IMAGE_SIZE: $image_size"
                echo "  VAL SET: $val_set"
                echo "========================================"
                python eval.py --backbone_arch $backbone --agg_arch $agg_arch --image_size $image_size $image_size --batch_size $batch_size --num_workers $num_workers --val_set_names $val_set --silent True
            done
        done
    done
done


