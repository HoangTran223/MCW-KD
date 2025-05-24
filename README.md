# MCW-KD: Multi-Cost Wasserstein Knowledge Distillation for Large Language Models

## 🔍 Overview
Multi-Cost Wasserstein Knowledge Distillation (MCW-KD), a novel framework that enhances KD by simultaneously optimizing several cost functions within a unified OT formulation. This repo provides code for reproducing our experiments.

## 🔧 Installation


1. **Create and activate the conda environment**:

   ```bash
   conda env create -f environment.yml
   conda activate MCW_KD
   pip install -r requirements.txt
   ```


# Training

### For MCW-KD:
For GPT2-120M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt2/multicost_gpt2_base.sh
```

For GPT2-120M with teacher is GPT2-1.5B, run:
```bash
bash scripts/gpt2_gpt2/multicost_gpt2_base.sh
```

For TinyLLaMA-1.1B with teacher is Mistral-7B, run:
```bash
bash scripts/tinyllama/multicost_tinyllama.sh
```

For GPT2-1.5B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/gptxl/multicost_gptxl.sh
```

### Baseline: Multi-Level Optimal Transport

For GPT2-120M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt2/multilevel_ot_gpt2_base.sh
```

For GPT2-120M with teacher is GPT2-1.5B, run:
```bash
bash scripts/gpt2_gpt2/multilevel_ot_gpt2_base.sh
```

For TinyLLaMA-1.1B with teacher is Mistral-7B, run:
```bash
bash scripts/tinyllama/multilevel_ot_tinyllama.sh
```

For GPT2-1.5B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/gptxl/multilevel_ot_gptxl.sh
```

### Baseline: Dual-Space KD with CMA

For GPT2-120M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt2/dskd_cma_gpt2_base.sh
```

For GPT2-120M with teacher is GPT2-1.5B, run:
```bash
bash scripts/gpt2_gpt2/dskd_cma_gpt2_base.sh
```

For TinyLLaMA-1.1B with teacher is Mistral-7B, run:
```bash
bash scripts/tinyllama/dskd_cma_tinyllama.sh
```

For GPT2-1.5B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/gptxl/dskd_cma_gptxl.sh
```

### Baseline: Logits Alignment by Minimum Edit Distance 

For GPT2-120M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt2/minedit_gpt2_base.sh
```

For GPT2-120M with teacher is GPT2-1.5B, run:
```bash
bash scripts/gpt2_gpt2/minedit_gpt2_base.sh
```

For TinyLLaMA-1.1B with teacher is Mistral-7B, run:
```bash
bash scripts/tinyllama/minedit_tinyllama.sh
```

For GPT2-1.5B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/gptxl/minedit_gptxl.sh
```

### Baseline: Universal Logit Distillation 

For GPT2-120M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt2/uld_gpt2_base.sh
```

For GPT2-120M with teacher is GPT2-1.5B, run:
```bash
bash scripts/gpt2_gpt2/uld_gpt2_base.sh
```

For TinyLLaMA-1.1B with teacher is Mistral-7B, run:
```bash
bash scripts/tinyllama/uld_tinyllama.sh
```

For GPT2-1.5B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/gptxl/uld_gptxl.sh
```

### SFT for student models
For GPT2-base (full fine-tuning), run:
```bash
bash scripts/gpt2/sft_gpt2_base.sh
```

For TinyLLaMA-1.1B (LoRA), run:
```bash
bash scripts/tinyllama/sft_tinyllama.sh
```

For GPT2-1.5B (LoRA), run:
```bash
bash scripts/gptxl/sft_gptxl.sh
```

## Evaluation
### Evaluate Full Fine-tuning Checkpoints
```bash
bash scripts/eval/run_eval.sh ${CKPT_PATH} ${EVAL_BATCH_SIZE}
```
According to this structure, `CKPT_PATH` is the **absolute path** of the model files.

### Evaluate LoRA Fine-tuning Checkpoints
```bash
bash scripts/eval/run_eval_lora.sh ${LORA_ADAPTER_PATH} ${EVAL_BATCH_SIZE}
```

Similarly, `LORA_ADAPTER_PATH` is the **absolute path** of the LoRA adapter.


## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


