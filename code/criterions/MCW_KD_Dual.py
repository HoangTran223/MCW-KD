import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import editdistance

from .various_divergence import VariousDivergence
from .ETP_dual import (
    MultiCostSemiDualOT,
    DualPotentialNetwork,
    SinkhornOT,
    pairwise_euclidean_distance,
    pairwise_cosine_distance
)


class MCW_KD_Dual(VariousDivergence):
    """
    Key implementation details matching the paper:
    1. φ is a LEARNABLE single-layer network
    2. φ^c is computed via c-transform formula
    3. α = softmax(β) with β updated via gradient descent
    4. Inner loop updates φ for I iterations
    """
    
    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        print("=" * 60)
        print("Using MCW-KD with ε-Entropic Semi-Dual Form (Paper Algorithm)")
        print("=" * 60)
        
        self.args = args
        
        # Precision settings
        if torch.cuda.is_available() and args.precision == "bf16":
            self.dtype = torch.bfloat16
        elif torch.cuda.is_available() and args.precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.window_size = 4
        self.padding_id = padding_id
        self.kd_rate = args.kd_rate
        self.tau_seq = 2.0
        self.top_k_vocab = getattr(args, 'top_k_vocab', 300)
        self.total_steps = args.total_iters
        self.current_step = 0
        
        self.epsilon = getattr(args, 'ot_epsilon', 0.1)
        self.num_inner_iters = getattr(args, 'ot_inner_iters', 4) 
        self.inner_lr = getattr(args, 'ot_inner_lr', 0.01)
        self.ot_weight_logits = getattr(args, 'ot_weight_logits', 1.0)
        self.ot_weight_hidden = getattr(args, 'ot_weight_hidden', 1.0)
        
        d_s = args.hidden_dim_student
        d_t = args.hidden_dim_teacher
        
        self.salience_proj_s = nn.Linear(d_s, 1, bias=True).to(self.device, dtype=self.dtype)
        
        self.mcw_ot_logits = MultiCostSemiDualOT(
            num_costs=3,
            input_dim_x=self.top_k_vocab,
            input_dim_y=self.top_k_vocab,
            hidden_dim=256,
            epsilon=self.epsilon,
            num_inner_iters=self.num_inner_iters,
            inner_lr=self.inner_lr
        ).to(self.device, dtype=torch.float32)  
        

        self.mcw_ot_hidden = MultiCostSemiDualOT(
            num_costs=3,
            input_dim_x=d_s, 
            input_dim_y=d_s, 
            hidden_dim=256,
            epsilon=self.epsilon,
            num_inner_iters=self.num_inner_iters,
            inner_lr=self.inner_lr
        ).to(self.device, dtype=torch.float32)
        
        print(f"MCW-KD Semi-Dual Config:")
        print(f"  - epsilon: {self.epsilon}")
        print(f"  - num_inner_iters (I): {self.num_inner_iters}")
        print(f"  - inner_lr: {self.inner_lr}")
        print(f"  - kd_rate: {self.kd_rate}")
        print(f"  - top_k_vocab: {self.top_k_vocab}")
    
    def forward(
        self,
        distiller,
        input_data,
        output_data,
        logging_output,
        batch_denom,
    ):
        self.current_step += 1
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        self.distiller.input_data = input_data
        
        # Teacher forward (frozen)
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None),
                output_hidden_states=True
            )
        
        # Student forward
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True
        )
        
        logits = outputs.logits
        log = {}
        
        # Cross-entropy loss
        loss_ce = self.compute_cross_entropy_loss(outputs.logits, output_data["label"], log=log)[0]
        log["loss_ce"] = loss_ce
        
        # Hidden states
        hidden_state_student = outputs.hidden_states[-1]
        hidden_state_teacher = teacher_outputs.hidden_states[-1]
        
        # Masks
        pad_mask = input_data["attention_mask"].bool()
        teacher_pad_mask = input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"].bool()
        
        ot_loss_logits, log = self.compute_mcw_ot_logits_semidual(
            distiller, outputs.logits, teacher_outputs.logits,
            pad_mask, teacher_pad_mask,
            outputs.hidden_states[-1], teacher_outputs.hidden_states[-1],
            log
        )
        
        ot_loss_hidden, log = self.compute_mcw_ot_hidden_semidual(
            distiller, outputs.hidden_states[-1], teacher_outputs.hidden_states[-1],
            pad_mask, teacher_pad_mask,
            log
        )
        
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        log["kd_loss"] = kd_loss
        
        # Total loss
        total_loss = 0.5 * loss_ce + self.kd_rate * kd_loss + 10 * ot_loss_logits + 10 * ot_loss_hidden
        log["loss"] = total_loss
        log["ot_loss_logits"] = ot_loss_logits
        log["ot_loss_hidden"] = ot_loss_hidden
        
        # Accuracy
        accuracy = self.compute_token_accuracy(logits, output_data["label"])
        log["accuracy"] = accuracy
        
        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        
        alpha_logits = self.mcw_ot_logits.alpha.detach().cpu().tolist()
        alpha_hidden = self.mcw_ot_hidden.alpha.detach().cpu().tolist()
        logging_output.setdefault("alpha_logits", []).append(alpha_logits)
        logging_output.setdefault("alpha_hidden", []).append(alpha_hidden)
        return total_loss / batch_denom, logging_output
    
    def compute_mcw_ot_logits_semidual(
        self,
        distiller,
        student_logits,
        teacher_logits,
        student_mask,
        teacher_mask,
        student_hiddens,
        teacher_hiddens,
        log,
        t_start=0.1,
        t_end=1.0
    ):

        batch_size = student_logits.size(0)
        tau = self.tau_seq
        eps = 1e-7
        k = self.top_k_vocab
        
        # Get top-k logits
        student_topk_logits, _ = student_logits.sort(dim=-1, descending=True)
        teacher_topk_logits, _ = teacher_logits.sort(dim=-1, descending=True)
        student_topk_logits = student_topk_logits[..., :k]
        teacher_topk_logits = teacher_topk_logits[..., :k]
        
        frac = min(self.current_step / self.total_steps, 1.0)
        t = t_start + (t_end - t_start) * frac
        interpolated_teacher_logits = (1 - t) * student_topk_logits + t * teacher_topk_logits
        
        student_probs = F.softmax(student_topk_logits / tau, dim=-1)
        teacher_probs = F.softmax(interpolated_teacher_logits / tau, dim=-1)
        
        total_loss = 0
        avg_W_list = [0.0, 0.0, 0.0]
        
        for b in range(batch_size):
            mask_s = student_mask[b].bool()
            mask_t = teacher_mask[b].bool()
            
            if mask_s.sum() == 0 or mask_t.sum() == 0:
                continue

            x = torch.clamp(teacher_probs[b][mask_t].float(), min=eps, max=1.0)  # (M, k)
            y = torch.clamp(student_probs[b][mask_s].float(), min=eps, max=1.0)  # (N, k)
            
            M, N = x.size(0), y.size(0)
            
            # Cost Matrix 1
            C1 = torch.cdist(x, y, p=2)
            C1_max = C1.max()
            C1 = C1 / (C1_max + eps) if C1_max > eps else C1
            
            # Cost Matrix 2
            x_safe = torch.clamp(x, min=eps)
            y_safe = torch.clamp(y, min=eps)
            log_ratio = torch.log(x_safe.unsqueeze(1) / y_safe.unsqueeze(0))
            C2 = (x_safe.unsqueeze(1) * log_ratio).sum(dim=-1)
            C2 = torch.clamp(C2, min=0, max=100)  # Clamp to reasonable range
            C2_max = C2.max()
            C2 = C2 / (C2_max + eps) if C2_max > eps else C2
            
            # Cost Matrix 3
            student_seq = student_hiddens[b][mask_s]
            teacher_seq = distiller.projectors["ot"](teacher_hiddens[b])[mask_t]
            sal_s = torch.sigmoid(self.salience_proj_s(student_seq.to(self.dtype))).squeeze(-1)
            sal_t = torch.sigmoid(self.salience_proj_s(teacher_seq.to(self.dtype))).squeeze(-1)
            C3 = torch.abs(sal_t.unsqueeze(1) - sal_s.unsqueeze(0)).float()
            C3_max = C3.max()
            C3 = C3 / (C3_max + eps) if C3_max > eps else C3
            
            C1 = torch.where(torch.isfinite(C1), C1, torch.zeros_like(C1))
            C2 = torch.where(torch.isfinite(C2), C2, torch.zeros_like(C2))
            C3 = torch.where(torch.isfinite(C3), C3, torch.zeros_like(C3))
            
            cost_matrices = [C1, C2, C3]
            
            L_MCW, W_list = self.mcw_ot_logits(
                x, y, cost_matrices,
                mu=None,  
                nu=None  
            )
            
            if torch.isfinite(L_MCW):
                total_loss += L_MCW
            for i, W_k in enumerate(W_list):
                if torch.isfinite(W_k):
                    avg_W_list[i] += W_k.item()
        
        if isinstance(total_loss, torch.Tensor) and torch.isfinite(total_loss):
            loss = total_loss * self.ot_weight_logits / batch_size
        else:
            loss = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True)
        
        for i, avg_W in enumerate(avg_W_list):
            log[f"W_{i+1}_logits"] = avg_W / max(batch_size, 1)
        log["ot_loss_logits"] = loss.item() if torch.isfinite(loss) else 0.0
        
        return loss, log
    
    def compute_mcw_ot_hidden_semidual(
        self,
        distiller,
        student_hiddens,
        teacher_hiddens,
        attention_mask_student,
        attention_mask_teacher,
        log
    ):
        
        teacher_hiddens_proj = distiller.projectors["ot"](teacher_hiddens)
        batch_size = teacher_hiddens.size(0)
        total_loss = 0
        eps = 1e-7
        
        avg_W_list = [0.0, 0.0, 0.0]
        
        for b in range(batch_size):
            mask_t = attention_mask_teacher[b].bool()
            mask_s = attention_mask_student[b].bool()
            
            if mask_s.sum() == 0 or mask_t.sum() == 0:
                continue
            
            x_raw = teacher_hiddens_proj[b][mask_t].float()  
            y_raw = student_hiddens[b][mask_s].float()
            
            x = F.normalize(x_raw, p=2, dim=-1)
            y = F.normalize(y_raw, p=2, dim=-1)
            
            M, N = x.size(0), y.size(0)
            
            student_ids = self.distiller.input_data["input_ids"][b][mask_s].tolist()
            teacher_ids = self.distiller.input_data[f"teacher_{distiller.teacher_model_type}_input_ids"][b][mask_t].tolist()
            stu_tok = distiller.student_tokenizer.convert_ids_to_tokens(student_ids, skip_special_tokens=True)
            tea_tok = distiller.teacher_tokenizers[distiller.teacher_model_type].convert_ids_to_tokens(teacher_ids, skip_special_tokens=True)
            
            # Cost Matrix 1
            C1 = self._compute_dtw_cost_matrix(stu_tok, tea_tok, M, N, x.device)
            C1_max = C1.max()
            C1 = C1 / (C1_max + eps) if C1_max > eps else C1
            
            # Cost Matrix 2
            ctx_x = self._compute_context_representations(x, self.window_size)
            ctx_y = self._compute_context_representations(y, self.window_size)

            C2 = torch.cdist(ctx_x, ctx_y, p=2)
            C2_max = C2.max()
            C2 = C2 / (C2_max + eps) if C2_max > eps else C2
            
            # Cost Matrix 3
            ctx_x_norm = F.normalize(ctx_x, p=2, dim=-1, eps=eps)
            ctx_y_norm = F.normalize(ctx_y, p=2, dim=-1, eps=eps)
            cosine_sim = torch.mm(ctx_x_norm, ctx_y_norm.T)
            C3 = 1 - cosine_sim
            C3 = torch.clamp(C3, min=0, max=2)  
            
            C1 = torch.where(torch.isfinite(C1), C1, torch.zeros_like(C1))
            C2 = torch.where(torch.isfinite(C2), C2, torch.zeros_like(C2))
            C3 = torch.where(torch.isfinite(C3), C3, torch.zeros_like(C3))
            
            cost_matrices = [C1, C2, C3]
        
            L_MCW, W_list = self.mcw_ot_hidden(
                x, y, cost_matrices,
                mu=None,
                nu=None
            )
            
            if torch.isfinite(L_MCW):
                total_loss += L_MCW
            for i, W_k in enumerate(W_list):
                if torch.isfinite(W_k):
                    avg_W_list[i] += W_k.item()
        
        if isinstance(total_loss, torch.Tensor) and torch.isfinite(total_loss):
            loss = total_loss * self.ot_weight_hidden / batch_size
        else:
            loss = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True)

        for i, avg_W in enumerate(avg_W_list):
            log[f"W_{i+1}_hidden"] = avg_W / max(batch_size, 1)
        log["ot_loss_hidden"] = loss.item() if torch.isfinite(loss) else 0.0
        
        return loss, log
    
    def _compute_dtw_cost_matrix(self, stu_tok, tea_tok, M, N, device):
        """Compute cost matrix based on DTW edit distance."""
        C = torch.zeros((M, N), device=device, dtype=torch.float32)
        
        edit_cache = {}
        def safe_edit(a, b):
            key = (a, b)
            if key not in edit_cache:
                edit_cache[key] = editdistance.eval(a, b)
            return edit_cache[key]
        
        pairs_s2t = dtw_alignment(stu_tok, tea_tok, dist_fn_edit)
        pairs_t2s = dtw_alignment(tea_tok, stu_tok, dist_fn_edit)
        
        C_s2t = torch.zeros((N, M), device=device, dtype=torch.float32)
        for i, j in pairs_s2t:
            if i < N and j < M:
                C_s2t[i, j] = safe_edit(stu_tok[i] if i < len(stu_tok) else "", 
                                         tea_tok[j] if j < len(tea_tok) else "")
        
        C_t2s = torch.zeros((M, N), device=device, dtype=torch.float32)
        for j, i in pairs_t2s:
            if j < M and i < N:
                C_t2s[j, i] = safe_edit(tea_tok[j] if j < len(tea_tok) else "",
                                         stu_tok[i] if i < len(stu_tok) else "")
        
        C = (C_s2t.T + C_t2s) / 2
        return C
    
    def _compute_context_representations(self, seq, window_size):
        """Compute context-aware representations using sliding window."""
        ctx = torch.zeros_like(seq)
        seq_len = seq.size(0)
        
        for i in range(seq_len):
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, seq_len)
            ctx[i] = seq[start:end].mean(dim=0)
        
        return ctx
    
    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        """Dual-Space KD with Cross-Model Attention (unchanged from original)."""
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)
        
        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]
        
        def get_embed_tokens(model):
            if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                return get_embed_tokens(model.base_model.model)
            if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "embed_tokens"):
                return model.model.decoder.embed_tokens
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                return model.model.embed_tokens
            if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                return model.transformer.wte
            if hasattr(model, "module"):
                return get_embed_tokens(model.module)
            raise NotImplementedError(f"Cannot find embed_tokens for model: {type(model)}")
        
        stu_embed_tokens = get_embed_tokens(distiller.student_model)
        tea_embed_tokens = get_embed_tokens(distiller.teacher_model)
        
        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        formal_input = torch.where(pad_mask, input_data["input_ids"], torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()
        
        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(
            teacher_pad_mask,
            input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
            torch.zeros_like(teacher_target)
        )
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()
        
        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)
        
        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()
        
        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()
        
        stu_v_hiddens = distiller.projectors["s2t"](hiddens).float()
        tea_v_hiddens = distiller.projectors["t2s"](
            norm_teacher_hiddens + norm_tea_target_embeds
        ).float()
        
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)
        
        t2s_weight = torch.softmax(align, -1)
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )
        
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum()
        max_probs = (t2s_logits.softmax(-1).max(-1)[0] * pad_mask).sum()
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_acc"] = t2s_acc
        log["max_t2s_prob"] = max_probs
        
        if not self.args.only_save_projector:
            t2s_kd_loss = self.dist_func(
                outputs.logits, t2s_logits.detach(), target, reduction="none", use_tea_temp=True
            )
            t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum()
            
            s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
            s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(hiddens)
            s2t_logits = s2t_hiddens.matmul(
                distiller.teacher_model.lm_head.weight.detach().transpose(-1, -2)
            )
            
            s2t_kd_loss = self.compute_forward_kl_divergence(
                s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
            )
            s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum()
            s2t_acc = (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum() * pad_mask.sum() / teacher_pad_mask.sum()
            
            kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
            log["t2s_kd_loss"] = t2s_kd_loss
            log["s2t_kd_loss"] = s2t_kd_loss
            log["s2t_acc"] = s2t_acc
        else:
            kd_loss = t2s_ce_loss
        
        log["kd_loss"] = kd_loss
        return kd_loss, log


def dist_fn_edit(a, b):
    """Edit distance function for DTW."""
    return editdistance.eval(a, b)


def dtw_alignment(series_1, series_2, norm_func=dist_fn_edit):
    """DTW alignment returning list of (i, j) pairs."""
    n1, n2 = len(series_1), len(series_2)
    if n1 == 0 or n2 == 0:
        return []
    
    matrix = np.zeros((n1 + 1, n2 + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(
                matrix[i, j + 1],
                matrix[i + 1, j],
                matrix[i, j]
            )
    
    # Backtrack
    matrix = matrix[1:, 1:]
    i, j = n1 - 1, n2 - 1
    aligned = []
    
    while i > 0 or j > 0:
        aligned.append((i, j))
        options = [
            matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
            matrix[i - 1, j] if i > 0 else np.inf,
            matrix[i, j - 1] if j > 0 else np.inf,
        ]
        move = np.argmin(options)
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    
    aligned.append((0, 0))
    return aligned
