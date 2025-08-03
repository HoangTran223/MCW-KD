import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .cross_entropy_loss import CrossEntropyLoss

def normalize(value):
    means = value.mean(dim=-1, keepdim=True)
    stds = value.std(dim=-1, keepdim=True)
    z_score_normalized_student = (value) / (stds + 0.0001)
    return z_score_normalized_student

def improved_sort(value):
    sums = value.sum(dim=(0, 1))
    sorted_indices = torch.argsort(sums, descending=True)
    sorted_values = value[:, :, sorted_indices]
    return sorted_values

def KL_wo(y_s, y_t, T=1):
    p_s = F.log_softmax(y_s / T, dim=-1)
    p_t = F.softmax(y_t / T, dim=-1)
    loss = -torch.sum(p_t * p_s, dim=-1).mean()
    return loss

class Sinkhorn_seq(nn.Module):
    def __init__(self, T=2):
        super(Sinkhorn_seq, self).__init__()
        self.T = T

    def sinkhorn_normalized(self, x, n_iters=20):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=10):
        Wxy = torch.cdist(x, y, p=1)
        K = torch.exp(-Wxy / epsilon)
        P = self.sinkhorn_normalized(K, n_iters)
        return torch.sum(P * Wxy)

    def forward(self, y_s, y_t):
        softmax = nn.Softmax(dim=-1)
        p_s = softmax(y_s / self.T)
        p_t = softmax(y_t / self.T)
        emd_loss = 0
        for i in range(p_s.shape[0]):
            emd_loss += 0.001 * self.sinkhorn_loss(x=p_s[i], y=p_t[i])
        return emd_loss

def greedy_algorithm_adjust_s(t, s):
    batch_size, T, k = t.shape
    _, n, _ = s.shape
    
    s_adjusted = torch.zeros_like(t)
    
    for b in range(batch_size):
        available_indices = list(range(n))
        
        for i in range(T):
            C_min = float('inf')
            j_star = -1
            
            for j in available_indices:
                # Compute cost as the sum of absolute differences for each batch
                C = torch.sum(torch.abs(t[b, :, i] - s[b, :, j]))
                
                if C < C_min:
                    C_min = C
                    j_star = j
            
            # Assign the best matching source vector to the adjusted tensor
            s_adjusted[b, :, i] = s[b, :, j_star]
            
            # Remove the selected index from available indices
            available_indices.remove(j_star)

    return s_adjusted

class MultiLevelOT(CrossEntropyLoss):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        print(f"-------------Using MultiLevelOT-------------")

        self.student_temperature = 2.0
        self.teacher_temperature = 2.0
        self.f = 1
        self.kd_rate = args.kd_rate

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom,
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        loss_ce = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True
            )
        
        kd_loss, log = self.compute_multi_level_ot_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )

        total_loss = loss_ce + self.kd_rate * kd_loss

        log["loss"] = total_loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"],
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return total_loss / batch_denom, logging_output


    def compute_multi_level_ot_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        student_target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        student_logits = outputs.logits
        teacher_logits = teacher_outputs.logits

        student_logits = normalize(student_logits)
        teacher_logits = normalize(teacher_logits)

        for i in range(student_target.shape[0]):
            stu_start_idx = student_target[i].ne(self.padding_id).nonzero()[0][0]
            tea_start_idx = teacher_target[i].ne(self.padding_id).nonzero()[0][0]
            student_target[i] = torch.cat([
                student_target[i][stu_start_idx:],
                student_target[i][:stu_start_idx]], dim=0
            )
            student_logits[i] = torch.cat([
                student_logits[i][stu_start_idx:, :],
                student_logits[i][:stu_start_idx, :]], dim=0
            )
            teacher_target[i] = torch.cat([
                teacher_target[i][tea_start_idx:],
                teacher_target[i][:tea_start_idx]], dim=0
            )
            teacher_logits[i] = torch.cat([
                teacher_logits[i][tea_start_idx:, :],
                teacher_logits[i][:tea_start_idx, :]], dim=0
            )

        student_probs = torch.softmax(student_logits / self.student_temperature, dim=-1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits / self.teacher_temperature, dim=-1, dtype=torch.float32)

        student_probs = improved_sort(student_probs)
        teacher_probs = improved_sort(teacher_probs)

        top_k = 50
        student_probs = student_probs[:, :, :top_k]
        teacher_probs = teacher_probs[:, :, :top_k]

        diff_size = student_probs.size(2) - teacher_probs.size(2)
        if diff_size > 0:
            teacher_probs = F.pad(teacher_probs, (0, diff_size), value=0)
        elif diff_size < 0:
            student_probs = F.pad(student_probs, (0, abs(diff_size)), value=0)

        sinkhorn = Sinkhorn_seq()
        if self.f == 1:
            ot_loss = sinkhorn(student_probs, teacher_probs) * 0.1
        elif self.f == 2:
            adjusted_student_probs = greedy_algorithm_adjust_s(teacher_probs, student_probs)
            ot_loss = sinkhorn(adjusted_student_probs, teacher_probs) * 0.1
        else:
            raise ValueError("Invalid value for f. Use 1 or 2.")

        kl_loss = KL_wo(student_probs, teacher_probs) * 0.1

        multi_loss = ot_loss + kl_loss + (student_probs - teacher_probs).abs().sum(-1).mean() 
        pad_mask = student_target.ne(self.padding_id) & teacher_target.ne(self.padding_id)
        multi_loss = (multi_loss * pad_mask).sum()

        log["loss"] = multi_loss
        return multi_loss, log