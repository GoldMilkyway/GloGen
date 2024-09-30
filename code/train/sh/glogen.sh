export CUDA_VISIBLE_DEVICES=0
# example) pre-trained: BCG / target: UCI2 / 5-shot
TRANS="bcg"
TARGET="uci2"
RESULTS="results_glogen"
SHOT=5
for EPOCH in 100
do
for LR in 1e-3
do
        for WD in 1e-4 0 1e-3
        do
            for GLO_COEF in 1e-2 1.0 10.0 
            do
                for GEN_COEF in 1e-2 1.0 10.0
                do
		            for VAR in 1e-4 1e-2 0 1 10
		            do
                        for SSVAR in 100 10 1 10000 1000 0.1
                        do
                        # glogen
                        python train.py --config_file ./core/config/dl/resnet/resnet_${TARGET}.yaml --result_dir ${RESULTS}/few_shot/pre_${TRANS}/glogen --backbone resnet1d --method prompt_glogen --lr $LR --wd $WD --gen_coeff $GEN_COEF --global_coeff $GLO_COEF --glonorm --clip --max_epochs $EPOCH --var $VAR --seed 0 --transfer $TRANS --shots $SHOT --ssvar $SSVAR --gvar --group_avg --remove &
                        sleep 5

                        # glogen PCA
                        python train.py --config_file ./core/config/dl/resnet/resnet_${TARGET}.yaml --result_dir ${RESULTS}/few_shot/pre_${TRANS}/glogen --backbone resnet1d --method prompt_glogen --lr $LR --wd $WD --gen_coeff $GEN_COEF --global_coeff $GLO_COEF --glonorm --clip --max_epochs $EPOCH --var $VAR --seed 0 --transfer $TRANS --shots $SHOT --ssvar $SSVAR --gvar --group_avg --remove --pca_encoding &
                        sleep 5

                        # glogen Trigger
                        python train.py --config_file ./core/config/dl/resnet/resnet_${TARGET}.yaml --result_dir ${RESULTS}/few_shot/pre_${TRANS}/glogen --backbone resnet1d --method prompt_glogen --lr $LR --wd $WD --gen_coeff $GEN_COEF --global_coeff $GLO_COEF --glonorm --clip --max_epochs $EPOCH --var $VAR --seed 0 --transfer $TRANS --shots $SHOT --ssvar $SSVAR --gvar --group_avg --remove --trigger &
                        sleep 5
                        done
            done
            wait
            done
            done
    done
done
done

