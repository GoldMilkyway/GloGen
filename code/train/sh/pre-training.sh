SEED=0

# Pre-training
for EXP in "bcg" "sensors" "uci2" "mimimc" "vitaldb"
do
for LR in 1e-4 1e-3 1e-5
do
    for WD in 0 1e-3 1e-4
    do
        export CUDA_VISIBLE_DEVICES=0
        python train.py --config_file ./core/config/dl/resnet/resnet_vitaldb.yaml --result_dir results/original --backbone resnet1d --method original --lr $LR --wd $WD --seed $SEED --save_checkpoint &
        sleep 20
    done
done
done
