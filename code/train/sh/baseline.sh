export CUDA_VISIBLE_DEVICES=0
# example) pre-trained: BCG / target: Sensors / 5-shot
TRANS="bcg"
TARGET="sensors"
SHOT=5
for EPOCH in 100
do
for LR in 1e-5
do
for WD in 1e-4 0 1e-3
do
            # Training From Scratch (No Pre-training) 
            python train.py --config_file ./core/config/dl/resnet/resnet_${TARGET}.yaml --result_dir results_baseline/few_shot/pre_${TRANS}/scratch --backbone resnet1d --method original --lr $LR --wd $WD --seed 0 --max_epochs $EPOCH --transfer $TRANS --shots $SHOT --scratch &
            sleep 5
            
            # Linear Probing
            python train.py --config_file ./core/config/dl/resnet/resnet_${TARGET}.yaml --result_dir results_baseline/few_shot/pre_${TRANS}/lp --backbone resnet1d --method original --lr $LR --wd $WD --seed 0 --max_epochs $EPOCH --transfer $TRANS --shots $SHOT --lp &
            sleep 5

            # Fine-tuning
            python train.py --config_file ./core/config/dl/resnet/resnet_${TARGET}.yaml --result_dir results_baseline/few_shot/pre_${TRANS}/ft --backbone resnet1d --method original --lr $LR --wd $WD --seed 0 --max_epochs $EPOCH --transfer $TRANS --shots $SHOT &
            sleep 5
done
done
wait
done
