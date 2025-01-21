#!/bin/bash

# Define variables
batch_size=128
num_workers=6
#val_set_names=("msls" "tokyo", "pitts30k")
#presets=("ResNet50_BoQ" "EigenPlaces" "CosPlaces" "CosPlacesR18D32" "CosPlacesR18D64" "CosPlacesR18D128" "CosPlacesR50D32" "CosPlacesR50D64" "CosPlacesR50D128" "EigenPlacesR18D256" "EigenPlacesR18D512" "EigenPlacesR50D128" "EigenPlacesR50D256" "EigenPlacesR50D512" "DinoV2_BoQ")

# Define the TeTRA models to evaluate
#backbone_archs=("vitsmallt" "vitbaset")
#agg_archs=("gem" "salad" "mixvpr" "boq")
#image_sizes=("224" "322")
#desc_divider_factors=("1" "2" "4")


batch_size=128
num_workers=8
val_set_names=("tokyo" "msls" "pitts30k")

presets=("EigenPlaces" "CosPlaces" "DinoSalad" "DinoV2_BoQ" "ResNet50_BoQ" "CosPlacesR18D32" "CosPlacesR18D128" "CosPlacesR50D32" "CosPlacesR50D64" "CosPlacesR50D128" "EigenPlacesR18D256" "EigenPlacesR18D512" "EigenPlacesR50D128" "EigenPlacesR50D256" "EigenPlacesR50D512")

backbone_archs=("vitbaset", "vitsmallt")
agg_archs=("boq" "salad" "mixvpr" "gem")
image_sizes=("322")
desc_divider_factors=("1" "2" "4")

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
        python eval.py --preset "$preset" --batch_size $batch_size --num_workers $num_workers --val_set_names $val_set --model_memory --dataset_retrieval_latency --dataset_descriptor_memory --dataset_total_memory --accuracy
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

#echo "================================================================================="
#echo "=============================== TeTRA models ==================================="
#echo "================================================================================="
#for backbone in "${backbone_archs[@]}"; do
#    for agg_arch in "${agg_archs[@]}"; do
#        for size in "${image_sizes[@]}"; do
#            for val_set in "${val_set_names[@]}"; do
#                for desc_divider in "${desc_divider_factors[@]}"; do
#                    echo "========================================"
#                    echo "Running evaluation for model:"
#                    echo "  BACKBONE_ARCH: $backbone"
#                    echo "  AGG_ARCH: $agg_arch"
#                    echo "  IMAGE_SIZE: $size x $size"
#                    echo "  VAL SET: $val_set"
#                    echo "  DESC_DIVIDER_FACTOR: $desc_divider"
#                    echo "========================================"
#                    python eval.py --backbone_arch $backbone --agg_arch $agg_arch --image_size "$size" "$size" \
#                        --batch_size $batch_size --num_workers $num_workers --val_set_names $val_set --desc_divider_factor $desc_divider --model_memory --dataset_retrieval_latency --dataset_descriptor_memory --dataset_total_memory --accuracy
#                done
#            done
#        done
#    done
#done



