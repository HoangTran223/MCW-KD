# MCW-KD: Multi-Cost Wasserstein Knowledge Distillation for Large Language Models

## 🔍 Overview
Multi-Cost Wasserstein Knowledge Distillation (MCW-KD), a novel framework that enhances KD by simultaneously optimizing several cost matrices within a unified OT formulation. This repo provides code for reproducing our experiments.

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
bash scripts/gpt2/MCW_KD.sh
```

For GPT2-340M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt_medium/MCW_KD.sh
```

For GPT2-120M with teacher is GPT2-1.5B, run:
```bash
bash scripts/gptxl_gpt2/MCW_KD.sh
```

For TinyLLaMA-1.1B with teacher is Mistral-7B, run:
```bash
bash scripts/tinyllama/MCW_KD.sh
```

For GPT2-1.5B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/gptxl/MCW_KD.sh
```

For OPT-2.7B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/opt/MCW_KD.sh
```


### Baseline: Multi-Level Optimal Transport for Universal Cross-Tokenizer (AAAI 2025)

For GPT2-120M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt2/MultiLevelOT.sh
```

For GPT2-340M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt_medium/MultiLevelOT.sh
```

For GPT2-120M with teacher is GPT2-1.5B, run:
```bash
bash scripts/gptxl_gpt2/MultiLevelOT.sh
```

For TinyLLaMA-1.1B with teacher is Mistral-7B, run:
```bash
bash scripts/tinyllama/MultiLevelOT.sh
```

For GPT2-1.5B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/gptxl/MultiLevelOT.sh
```

For OPT-2.7B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/opt/MultiLevelOT.sh
```

### Baseline: Dual-Space Knowledge Distillation (EMNLP 2024)

For GPT2-120M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt2/DSKD.sh
```

For GPT2-340M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt_medium/DSKD.sh
```

For GPT2-120M with teacher is GPT2-1.5B, run:
```bash
bash scripts/gptxl_gpt2/DSKD.sh
```

For TinyLLaMA-1.1B with teacher is Mistral-7B, run:
```bash
bash scripts/tinyllama/DSKD.sh
```

For GPT2-1.5B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/gptxl/DSKD.sh
```

For OPT-2.7B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/opt/DSKD.sh
```


### Baseline: Logits Alignment by Minimum Edit Distance (ICLR 2024)

For GPT2-120M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt2/MinED.sh
```

For GPT2-340M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt_medium/MinED.sh
```

For GPT2-120M with teacher is GPT2-1.5B, run:
```bash
bash scripts/gptxl_gpt2/MinED.sh
```

For TinyLLaMA-1.1B with teacher is Mistral-7B, run:
```bash
bash scripts/tinyllama/MinED.sh
```

For GPT2-1.5B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/gptxl/MinED.sh
```

For OPT-2.7B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/opt/MinED.sh
```

### Baseline: Universal Logit Distillation (TMLR 2025)

For GPT2-120M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt2/ULD.sh
```

For GPT2-340M with teacher is Qwen1.5, run:
```bash
bash scripts/gpt_medium/ULD.sh
```

For GPT2-120M with teacher is GPT2-1.5B, run:
```bash
bash scripts/gptxl_gpt2/ULD.sh
```

For TinyLLaMA-1.1B with teacher is Mistral-7B, run:
```bash
bash scripts/tinyllama/ULD.sh
```

For GPT2-1.5B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/gptxl/ULD.sh
```

For OPT-2.7B with teacher is Qwen2.5-7B-Instruct, run:
```bash
bash scripts/opt/ULD.sh
```

### SFT for Student Models
For GPT2-base (full fine-tuning), run:
```bash
bash scripts/gpt2/SFT.sh
```

For GPT2-base (full fine-tuning), run:
```bash
bash scripts/gpt_medium/SFT.sh
```

For TinyLLaMA-1.1B (LoRA), run:
```bash
bash scripts/tinyllama/SFT.sh
```

For GPT2-1.5B (LoRA), run:
```bash
bash scripts/gptxl/SFT.sh
```

For OPT-2.7B (LoRA), run:
```bash
bash scripts/opt/SFT.sh
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

### GPT-4 Evaluation
To perform evaluation with GPT-4, run:
```bash
python code/analysis/llm_judge
```

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


