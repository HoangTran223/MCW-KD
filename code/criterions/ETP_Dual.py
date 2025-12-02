
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DualPotentialNetwork(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        use_context: bool = True
    ):
        super().__init__()
        self.use_context = use_context
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        if use_context:
            self.context_proj = nn.Linear(input_dim, hidden_dim)
            self.token_proj = nn.Linear(input_dim, hidden_dim)
            self.combine = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: torch.Tensor = None
    ) -> torch.Tensor:

        if self.use_context and context is not None:
            if x.dim() == 2:
                ctx_feat = self.context_proj(context).unsqueeze(0).expand(x.size(0), -1)
                tok_feat = self.token_proj(x)
                combined = torch.cat([ctx_feat, tok_feat], dim=-1)
                output = torch.tanh(self.combine(combined).squeeze(-1))
                return output  # (M,)
            else:
                B, M, D = x.shape
                ctx_feat = self.context_proj(context).unsqueeze(1).expand(-1, M, -1)
                tok_feat = self.token_proj(x)
                combined = torch.cat([ctx_feat, tok_feat], dim=-1)
                output = torch.tanh(self.combine(combined).squeeze(-1))
                return output  # (B, M)
        else:
            return torch.tanh(self.network(x).squeeze(-1))


class EntropicSemiDualOT(nn.Module):

    def __init__(
        self,
        input_dim_x: int,
        input_dim_y: int,
        hidden_dim: int = 256,
        epsilon: float = 0.1,
        num_inner_iters: int = 5,
        inner_lr: float = 0.01,
        use_context: bool = True
    ):
        """
        Args:
            input_dim_x: Dimension of source (teacher) representations
            input_dim_y: Dimension of target (student) representations  
            hidden_dim: Hidden dimension for φ network
            epsilon: Entropic regularization parameter
            num_inner_iters: Number of iterations to update 
            inner_lr: Learning rate for inner loop φ updates
            use_context: Whether to use context-aware φ
        """
        super().__init__()
        
        self.epsilon = epsilon
        self.num_inner_iters = num_inner_iters
        self.inner_lr = inner_lr
        
        # Parameterized dual potential φ
        self.phi = DualPotentialNetwork(
            input_dim=input_dim_x,
            hidden_dim=hidden_dim,
            output_dim=1,
            use_context=use_context
        )
        
        # For projection if dimensions differ
        if input_dim_x != input_dim_y:
            self.proj_y = nn.Linear(input_dim_y, input_dim_x)
        else:
            self.proj_y = nn.Identity()
    
    def compute_c_transform(
        self,
        phi_x: torch.Tensor,
        C: torch.Tensor,
        mu: torch.Tensor
    ) -> torch.Tensor:

        eps = self.epsilon
        M, N = C.shape
        exponent = (phi_x.unsqueeze(1) - C) / eps  # (M, N)
        
        log_mu = torch.log(mu + 1e-10)  # (M,)
        exponent = exponent + log_mu.unsqueeze(1)  # (M, N)
        phi_c_y = -eps * torch.logsumexp(exponent, dim=0)  # (N,)
        
        return phi_c_y
    
    def compute_wasserstein(
        self,
        phi_x: torch.Tensor,
        phi_c_y: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor
    ) -> torch.Tensor:

        term1 = torch.dot(phi_x, mu)
        term2 = torch.dot(phi_c_y, nu)
        return term1 + term2
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        C: torch.Tensor,
        mu: torch.Tensor = None,
        nu: torch.Tensor = None,
        update_phi: bool = True
    ) -> torch.Tensor:

        M, D_x = x.shape
        N, D_y = y.shape
        device = x.device
        dtype = x.dtype

        if mu is None:
            mu = torch.ones(M, device=device, dtype=dtype) / M
        if nu is None:
            nu = torch.ones(N, device=device, dtype=dtype) / N

        context = x.mean(dim=0)  # (D_x,)
        
        if update_phi:
            for _ in range(self.num_inner_iters):
                phi_x = self.phi(x, context)  # (M,)
                phi_c_y = self.compute_c_transform(phi_x, C, mu)  # (N,)
                
                W_eps = self.compute_wasserstein(phi_x, phi_c_y, mu, nu)
                loss = -W_eps
                
                if self.training:
                    grads = torch.autograd.grad(
                        loss, 
                        self.phi.parameters(),
                        create_graph=True,
                        retain_graph=True
                    )
                    with torch.no_grad():
                        for param, grad in zip(self.phi.parameters(), grads):
                            param.data = param.data - self.inner_lr * grad
        
        phi_x = self.phi(x, context)
        phi_c_y = self.compute_c_transform(phi_x, C, mu)
        W_eps = self.compute_wasserstein(phi_x, phi_c_y, mu, nu)
        
        return W_eps


class MultiCostSemiDualOT(nn.Module):
    def __init__(
        self,
        num_costs: int,
        input_dim_x: int,
        input_dim_y: int,
        hidden_dim: int = 256,
        epsilon: float = 0.1,
        num_inner_iters: int = 5,
        inner_lr: float = 0.01
    ):
        
        super().__init__()
        
        self.num_costs = num_costs
        self.epsilon = epsilon
        self.phi = DualPotentialNetwork(
            input_dim=input_dim_x,
            hidden_dim=hidden_dim,
            output_dim=1,
            use_context=True
        )
        self.beta = nn.Parameter(torch.zeros(num_costs))
        self.num_inner_iters = num_inner_iters
        self.inner_lr = inner_lr

        if input_dim_x != input_dim_y:
            self.proj_y = nn.Linear(input_dim_y, input_dim_x)
        else:
            self.proj_y = nn.Identity()
    
    @property
    def alpha(self) -> torch.Tensor:
        return F.softmax(self.beta, dim=0)
    
    def compute_c_transform_multi(
        self,
        phi_x: torch.Tensor,
        cost_matrices: list,
        mu: torch.Tensor
    ) -> torch.Tensor:
        
        eps = self.epsilon
        alpha = self.alpha
        phi_c_list = []

        for k, C_k in enumerate(cost_matrices):
            M, N = C_k.shape
            
            phi_x_clamped = torch.clamp(phi_x, -50, 50)
            exponent = (phi_x_clamped.unsqueeze(1) - C_k) / eps
            exponent = torch.clamp(exponent, -50, 50)
            
            log_mu = torch.log(mu + 1e-10)
            exponent = exponent + log_mu.unsqueeze(1)

            phi_c_k = -eps * torch.logsumexp(exponent, dim=0)
            phi_c_k = torch.where(torch.isfinite(phi_c_k), phi_c_k, torch.zeros_like(phi_c_k))
            phi_c_list.append(phi_c_k)
        
        phi_c_stack = torch.stack(phi_c_list, dim=0)  # (K, N)
        phi_c_weighted = (alpha.unsqueeze(1) * phi_c_stack).sum(dim=0)  # (N,)
        
        return phi_c_weighted, phi_c_list
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cost_matrices: list,
        mu: torch.Tensor = None,
        nu: torch.Tensor = None
    ) -> tuple:
        
        assert len(cost_matrices) == self.num_costs
        
        M = x.size(0)
        N = y.size(0)
        device = x.device
        dtype = x.dtype
        
        if mu is None:
            mu = torch.ones(M, device=device, dtype=dtype) / M
        if nu is None:
            nu = torch.ones(N, device=device, dtype=dtype) / N
        
        context = x.mean(dim=0)
        for _ in range(self.num_inner_iters):
            phi_x = self.phi(x, context)

            if torch.isnan(phi_x).any():
                phi_x = torch.zeros_like(phi_x)
                break
            
            phi_c_y, _ = self.compute_c_transform_multi(phi_x, cost_matrices, mu)
            
            if torch.isnan(phi_c_y).any():
                break
            
            W_eps = torch.dot(phi_x, mu) + torch.dot(phi_c_y, nu)
            
            if torch.isnan(W_eps):
                break

            if self.training:
                loss = -W_eps
                try:
                    grads = torch.autograd.grad(
                        loss,
                        self.phi.parameters(),
                        create_graph=False, 
                        retain_graph=True,
                        allow_unused=True
                    )
                    with torch.no_grad():
                        for param, grad in zip(self.phi.parameters(), grads):
                            if grad is not None:
                                # Gradient clipping
                                grad_clipped = torch.clamp(grad, -1.0, 1.0)
                                param.data = param.data - self.inner_lr * grad_clipped
                except RuntimeError:
                    break
        
        phi_x = self.phi(x, context)
        if torch.isnan(phi_x).any():
            phi_x = torch.zeros_like(phi_x)
        
        phi_c_y, phi_c_list = self.compute_c_transform_multi(phi_x, cost_matrices, mu)
        
        L_MCW = torch.dot(phi_x, mu) + torch.dot(phi_c_y, nu)
        if torch.isnan(L_MCW):
            L_MCW = torch.tensor(0.0, device=x.device, dtype=x.dtype, requires_grad=True)
        
        W_list = []
        for phi_c_k in phi_c_list:
            W_k = torch.dot(phi_x, mu) + torch.dot(phi_c_k, nu)
            # Handle nan
            if torch.isnan(W_k):
                W_k = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            W_list.append(W_k)
        
        return L_MCW, W_list


class SinkhornOT(nn.Module):
    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 100,
        threshold: float = 1e-6
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.threshold = threshold
    
    def forward(
        self,
        C: torch.Tensor,
        mu: torch.Tensor = None,
        nu: torch.Tensor = None
    ) -> torch.Tensor:

        n, m = C.shape
        device = C.device
        dtype = C.dtype
        
        if mu is None:
            mu = torch.ones(n, device=device, dtype=dtype) / n
        if nu is None:
            nu = torch.ones(m, device=device, dtype=dtype) / m

        log_mu = torch.log(mu + 1e-10)
        log_nu = torch.log(nu + 1e-10)
        
        u = torch.zeros(n, device=device, dtype=dtype)
        v = torch.zeros(m, device=device, dtype=dtype)
        
        K = -C / self.epsilon
        
        for _ in range(self.max_iter):
            u_old = u.clone()
            u = log_mu - torch.logsumexp(K + v.unsqueeze(0) + log_nu.unsqueeze(0), dim=1)
            v = log_nu - torch.logsumexp(K + u.unsqueeze(1) + log_mu.unsqueeze(1), dim=0)
            
            if torch.max(torch.abs(u - u_old)) < self.threshold:
                break
        
        f = self.epsilon * u
        g = self.epsilon * v
        W_eps = torch.dot(f, mu) + torch.dot(g, nu)
        
        return W_eps

def pairwise_euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distance matrix."""
    return torch.cdist(x, y, p=2)

def pairwise_cosine_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:

    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    sim = torch.mm(x_norm, y_norm.T)
    return 1 - sim


def pairwise_kl_distance(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:

    p = p + eps
    q = q + eps
    log_p = torch.log(p)
    log_q = torch.log(q)
    entropy_p = (p * log_p).sum(dim=-1, keepdim=True)
    cross_entropy = torch.mm(p, log_q.T)
    return entropy_p - cross_entropy
