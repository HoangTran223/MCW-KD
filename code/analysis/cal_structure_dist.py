import os
import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import pyplot as plt
from tqdm import tqdm

save_path = "../outputs/picture"
if not os.path.exists(save_path):
    os.mkdir(save_path)

device = "cuda:0"  
device_teacher = "cuda:1"  

# Model paths
sft_path = ""
dskd_path = ""
uld_path = ""
mined_path = ""
multiot_path = ""
mcw_kd_path = ""
teacher_path = ""

print("Loading teacher model...")
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_path).to(device_teacher)
print(f"Teacher model loaded on {device_teacher}. GPU memory: {torch.cuda.memory_allocated(device_teacher)/1024**3:.2f}GB")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(sft_path)

with open("../data/dolly/train.jsonl") as f:
    data = [json.loads(s) for s in f.readlines()[:1000]]

print("Preparing data...")
prompt = [d["prompt"][:256] for d in data]
output = [d["output"][:256] for d in data]

prompt_inputs = [tokenizer(text, return_tensors="pt") for text in prompt]
output_inputs = [tokenizer(text, return_tensors="pt") for text in output]
print(f"Data prepared: {len(prompt_inputs)} samples")

def cal_all_sim(model, teacher_model):
    all_cosine_dist = []
    all_innerprod_dist = []
    print("Calculating similarities...")
    for i, (pinp, oinp) in enumerate(tqdm(list(zip(prompt_inputs, output_inputs)))):
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(prompt_inputs)}, GPU memory: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB")
        
        inp = {}
        for key in pinp:
            inp[key] = torch.cat([pinp[key], oinp[key]], 1)
        
        inp["position_ids"] = torch.tensor([list(range(inp["input_ids"].shape[1]))])

        for x in inp:
            inp[x] = inp[x].to(device)
        
        prompt_len = pinp["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inp, output_hidden_states=True)
            hiddens = outputs.hidden_states[-1][:, prompt_len:]
        
            # Move input to teacher device for teacher model inference
            inp_teacher = {}
            for key in inp:
                inp_teacher[key] = inp[key].to(device_teacher)
            
            teacher_outputs = teacher_model(**inp_teacher, output_hidden_states=True)
            teacher_hiddens = teacher_outputs.hidden_states[-1][:, prompt_len:].to(device)  # Move back to main device
            
        norm_hiddens = hiddens / hiddens.norm(p=2, dim=-1, keepdim=True)
        stu_self_cosine = norm_hiddens.matmul(norm_hiddens.transpose(-1, -2))
        stu_self_innerprod = hiddens.matmul(hiddens.transpose(-1, -2))
        stu_self_innerprod = stu_self_innerprod / stu_self_innerprod.sum(-1, keepdim=True)

        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.norm(p=2, dim=-1, keepdim=True)
        tea_self_cosine = norm_teacher_hiddens.matmul(norm_teacher_hiddens.transpose(-1, -2))
        tea_self_innerprod = teacher_hiddens.matmul(teacher_hiddens.transpose(-1, -2))
        tea_self_innerprod = tea_self_innerprod / tea_self_innerprod.sum(-1, keepdim=True)

        cosine_sim = (stu_self_cosine - tea_self_cosine).abs().mean()
        innerprod_sim = (stu_self_innerprod - tea_self_innerprod).abs().sum(-1).mean()
        all_cosine_dist.append(cosine_sim.cpu().item())
        all_innerprod_dist.append(innerprod_sim.cpu().item())

        # Clear intermediate tensors
        del inp, inp_teacher, outputs, hiddens, teacher_outputs, teacher_hiddens
        del norm_hiddens, stu_self_cosine, stu_self_innerprod
        del norm_teacher_hiddens, tea_self_cosine, tea_self_innerprod
        torch.cuda.empty_cache()

    return all_cosine_dist, all_innerprod_dist

def load_model_and_calculate(model_path, model_name):
    print(f"\n=== Processing {model_name} ===")
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    print(f"Model loaded on {device}. GPU memory: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB")
    print(f"Teacher on {device_teacher}. GPU memory: {torch.cuda.memory_allocated(device_teacher)/1024**3:.2f}GB")
    
    cosine_sim, innerprod_sim = cal_all_sim(model, teacher_model)
    
    print(f"Calculation complete for {model_name}")
    print(f"Freeing model memory...")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Memory after cleanup: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB")
    
    return cosine_sim, innerprod_sim

# Calculate similarities for all models one by one
print("\n" + "="*50)
print("Starting model evaluations...")
print("="*50)

model_paths = [
    (sft_path, "SFT"),
    (dskd_path, "DSKD"), 
    (uld_path, "ULD"),
    (mined_path, "MinED"),
    (multiot_path, "MultiLevelOT"),
    (mcw_kd_path, "MCW_KD")
]

all_cosine_results = []
all_innerprod_results = []
model_labels = []

for model_path, model_name in model_paths:
    cosine_sim, innerprod_sim = load_model_and_calculate(model_path, model_name)
    all_cosine_results.append(cosine_sim)
    all_innerprod_results.append(innerprod_sim)
    model_labels.append(model_name)

print("\n" + "="*50)
print("All calculations complete! Creating plots...")
print("="*50)

# Plot cosine similarity comparison
plt.boxplot(
    all_cosine_results, 
    labels=model_labels,
    showfliers=False,
    showmeans=False
)
plt.grid(axis="y", linestyle=":")
plt.xlabel("Methods")
plt.ylabel("Cosine as Structure")
plt.savefig(save_path + "/cosine.png")
plt.savefig(save_path + "/cosine.pdf")

plt.cla()
plt.boxplot(
    all_innerprod_results, 
    labels=model_labels, 
    showfliers=False,
    showmeans=False
)
plt.grid(axis="y", linestyle=":")
plt.ylabel(" Inner Product as Structure")
plt.savefig(save_path + "/attn.png")
plt.savefig(save_path + "/attn.pdf")

print("Plots saved successfully!")
print(f"GPU {device} memory at end: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB")
print(f"GPU {device_teacher} memory at end: {torch.cuda.memory_allocated(device_teacher)/1024**3:.2f}GB")