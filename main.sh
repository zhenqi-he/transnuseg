export PYTHON_PATH='./train.py'
export MODEL_TYPE='transnuseg'
export ALPHA='0.3'
export BETA='0.35'
export GAMMA='0.35'
export SHARING_RATIO='0.5'
export DATASET='Radiology'
export NUM_EPOCH=300
export LR=0.001
export RANDOM_SEED=666
export BATCH_SIZE=2

python $PYTHON_PATH --model_type=$MODEL_TYPE --alpha=$ALPHA --beta=$BETA --gamma=$GAMMA --sharing_ratio=$SHARING_RATIO --dataset=$DATASET --lr=$LR --num_epoch=$NUM_EPOCH --random_seed=$RANDOM_SEED --batch_size=$BATCH_SIZE
