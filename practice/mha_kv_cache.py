import torch
import math

from torch import nn

class MHA(nn.Module):

    def __init__(
        self,
        n_head,
        head_dim,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.W_q = nn.Linear(n_head*head_dim, n_head*head_dim)
        self.W_k = nn.Linear(n_head*head_dim, n_head*head_dim)
        self.W_v = nn.Linear(n_head*head_dim, n_head*head_dim)
        self.W_o = nn.Linear(n_head*head_dim, n_head*head_dim)
        self.k_cache = None
        self.v_cache = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, S, D = x.shape
        assert D == self.n_head * self.head_dim
        n_head, head_dim = self.n_head, self.head_dim
        Q = self.W_q(x).view(B, S, n_head, head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, S, n_head, head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, S, n_head, head_dim).transpose(1, 2)
        
        if self.k_cache is None:
                self.k_cache = torch.zeros(B, n_head, 0, head_dim)
                self.v_cache = torch.zeros(B, n_head, 0, head_dim)
        self.k_cache = torch.cat([self.k_cache, K], dim=2)
        self.v_cache = torch.cat([self.v_cache, V], dim=2)

        S_k = self.k_cache.shape[2]
        start_pos = S_k - S
        q_offset = torch.arange(S).unsqueeze(1) # [S, 1]
        q_pos = start_pos + q_offset
        k_pos = torch.arange(S_k).unsqueeze(0) # [1, S_k]
        mask = torch.where(q_pos >= k_pos, 0.0, float("-inf"))

        print(f"mask: {mask}\n")

        logits = Q @ self.k_cache.transpose(2, 3) / math.sqrt(head_dim)
        print(f"logits: {logits}\n")

        logits = logits + mask
        print(f"logits: {logits}\n")

        score = torch.softmax(logits, dim=2)
        x = score @ self.v_cache
        o = self.W_o(x.transpose(1, 2).contiguous().view(B, S, D))
        
        return o


if __name__ == "__main__":
    mha = MHA(n_head=2, head_dim=4)
    x = torch.randn(1, 3, 8)
    print(f"x: {x}\n")
    o = mha(x)
    print(f"o: {o}\n")

