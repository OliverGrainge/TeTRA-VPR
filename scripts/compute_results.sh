#!/bin/bash

# Define variables
batch_size=128
num_workers=12
val_set_names=("msls" "tokyo" "pitts30k")
#val_set_names=("msls" "tokyo")

#define the presets to evaluate
#presets=("CosPlaces" "DinoSalad" "EigenPlaces" "DinoV2_BoQ" "ResNet50_BoQ")
presets=("CosPlacesR18D32" "CosPlacesR18D64" "CosPlacesR18D128" "CosPlacesR50D32" "CosPlacesR50D64" "CosPlacesR50D128" "EigenPlacesR18D256" "EigenPlacesR18D512" "EigenPlacesR50D128" "EigenPlacesR50D256" "EigenPlacesR50D512" "EigenPlacesR50D1024")

# define the TeTRA models to evaluate
#backbone_archs=("vit_base_PLRBitLinear" "vit_small_PLRBitLinear")
#agg_archs=("salad" "gem" "boq" "mixvpr")
#image_size=("224" "322")
backbone_archs=("vit_small_PLRBitLinear" "vit_base_PLRBitLinear")
agg_archs=("gem" "salad" "mixvpr" "boq")
image_size=("224" "322")




echo "================================================================================="
echo "=============================== Baseline models ================================"
echo "================================================================================="
for preset in "${presets[@]}"; do
    for val_set in "${val_set_names[@]}"; do
        echo "========================================"#
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
#for backbone in "${backbone_archs[@]}"; do
#    for agg_arch in "${agg_archs[@]}"; do
#        for size in "${image_size[@]}"; do  # Fixed naming to avoid shadowing issues
#            for val_set in "${val_set_names[@]}"; do
#                echo "========================================"
#                echo "Running evaluation for model:"
#                echo "  BACKBONE_ARCH: $backbone"
#                echo "  AGG_ARCH: $agg_arch"
#                echo "  IMAGE_SIZE: $size x $size"
#                echo "  VAL SET: $val_set"
#                echo "========================================"
#                python eval.py --backbone_arch $backbone --agg_arch $agg_arch --image_size "$size" "$size" \
#                    --batch_size $batch_size --num_workers $num_workers --val_set_names $val_set --silent True
#            done
#        done
#    done
#done



