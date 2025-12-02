#!/bin/bash
GPUS=(0 1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"


BASE_PATH=path_to_project
CKPT_TYPE="name_student_model"
CKPT_NAME="name_student_model"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_TYPE}/${CKPT_NAME}"
TEACHER_MODEL_TYPE="name_teacher_model"
TEACHER_MODEL_NAME="name_teacher_model"
TEACHER_MODEL_PATH="path_to_teacher_sft_ckpt"

DATA_DIR="${BASE_PATH}/data/dolly/"
TASK="MCW_KD_Dual"
BATCH_SIZE=4
LR=0.001
GRAD_ACC=2
EVAL_BATCH_SIZE=16
EPOCH=15
KD_RATE=2
KD_TEMP=2.0
LORA_RANK=256
LORA_ALPHA=8
LORA_DROPOUT=0.1


OT_EPSILON_LOGITS=0.1      # Entropic regularization for logits OT
OT_EPSILON_HIDDEN=0.1      # Entropic regularization for hidden OT
OT_MAX_ITER=500            # Max Sinkhorn iterations in dual space
OT_THRESHOLD=1e-7          # Convergence threshold
OT_WEIGHT_LOGITS=1.0       # Weight for OT loss on logits
OT_WEIGHT_HIDDEN=1.0       # Weight for OT loss on hidden
WEIGHT_UPDATE_INTERVAL=80  # Steps between adaptive weight updates

HIDDEN_DIM_STUDENT=1280
HIDDEN_DIM_TEACHER=3584
PROJ_DIM=1280

MAX_LENGTH=512
PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=0.001

PRECISION="bf16"
CRITERION="MCW_KD_Dual"
KD_OBJ="forward_kl"

CONFIG="${KD_OBJ}-dual-lora-rank=${LORA_RANK}-alpha=${LORA_ALPHA}-dropout=${LORA_DROPOUT}-${PRECISION}"
SETTING="criterion=${CRITERION}__${CONFIG}__teacher=${TEACHER_MODEL_TYPE}__kd^rate=${KD_RATE}__kd^temp=${KD_TEMP}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}"
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_TYPE}/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=1

# Seed
SEED=10

mkdir -p ${SAVE_PATH}

OPTS=""

OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"
OPTS+=" --hidden-dim-student ${HIDDEN_DIM_STUDENT}"
OPTS+=" --hidden-dim-teacher ${HIDDEN_DIM_TEACHER}"
OPTS+=" --proj_dim ${PROJ_DIM}"


OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
OPTS+=" --task ${TASK}"

OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --num-epochs ${EPOCH}"
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --kd-objective ${KD_OBJ}"
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r ${LORA_RANK}"
OPTS+=" --peft-lora-alpha ${LORA_ALPHA}"
OPTS+=" --peft-lora-dropout ${LORA_DROPOUT}"

OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --projector-config-path ${PROJECTOR_CONFIG_PATH}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --precision ${PRECISION}"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"

OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
if [[ $PRECISION == "bf16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ $PRECISION == "fp16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ $PRECISION == "fp32" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi

OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}


echo "============================================================"
echo "MCW-KD Training"
echo "============================================================"
echo "Criterion: ${CRITERION}"
echo "Student: ${CKPT_PATH}"
echo "Teacher: ${TEACHER_MODEL_PATH}"
echo "OT Epsilon (logits): ${OT_EPSILON_LOGITS}"
echo "OT Epsilon (hidden): ${OT_EPSILON_HIDDEN}"
echo "Save path: ${SAVE_PATH}"
echo "============================================================"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/distillation_dual.py ${OPTS}"

${CMD} 2>&1 | tee ${SAVE_PATH}/train.log

echo "============================================================"
echo "Training completed!"
echo "Logs saved to: ${SAVE_PATH}/train.log"
echo "============================================================"

