export PYTHON_PATH='./train.py'
export MODEL_PATH='transnucseg'
export ALPHA='0.3'
export BETA='0.35'
export GAMMA='0.35'
export SHARING_RATIO='0.5'
export DATASET='Histology'

python $PYTHON_PATH --model_type=$MODEL_PATH --alpha=$ALPHA --beta=$BETA --gamma=$GAMMA --sharing_ratio=$SHARING_RATIO --dataset=$DATASET
