set -e

CKPT_PATH=$1
EVAL_BATCH_SIZE=${2-8}

WORK_DIR="dir"
GPUS="0"
GPUS_PER_NODE=1
MASTER_PORT=2000  

for DATASET in vicuna self-inst dolly vicuna "sinst/11_"; do
  for SEED in 10 20 30 40 50; do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh \
         "${GPUS}" \
         ${MASTER_PORT} \
         ${GPUS_PER_NODE} \
         ${WORK_DIR} \
         "${CKPT_PATH}" \
         "${DATASET}" \
         ${EVAL_BATCH_SIZE} \
         ${SEED}
    MASTER_PORT=$((MASTER_PORT+1))
  done
done