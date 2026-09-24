import dataclasses
import torch
from torch import nn
from torch.nn import functional as F
from typing import List

@dataclasses.dataclass
class MoeArgs():
    num_experts: int
    num_experts_per_token: int

class MoeLayer(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        num_experts_per_token: int,
        gate: nn.Module,
    ):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = MoeArgs(num_experts=len(experts), num_experts_per_token=num_experts_per_token)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, D)
        Returns:
            results: (N, D)
        """

        gate_logits = self.gate(x)
        weights, selected_experts = torch.topk(gate_logits, k=self.args.num_experts_per_token) # N, K
        weights = F.softmax(weights, dim=-1)

        print(f"weights: {weights}\n")
        print(f"selected_experts: {selected_experts}\n")

        results = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # selected_experts: (N, K)
            # e.g. selected_experts=[[2, 0], [1, 0]]
            # selected_experts =
            #     token0  [2, 0]
            #     token1  [1, 0]
            #     token2  [2, 3]
            # i = 0: nth_token=[0, 1], nth_expert=[1, 1]  (0,1) (1,1)
            # i = 1: nth_token=[1],    nth_expert=[0]     (1,0)
            # i = 2: nth_token=[0, 2], nth_expert=[0, 0]  (0,0) (2,0)
            # i = 3: nth_token=[2],    nth_expert=[1]     (2,1)
            print(f"expert_id: {i}")
            print(f"condition matrix: {selected_experts == i}")
            token_ids, expert_ids = torch.where(selected_experts == i)
            print(f"token_ids: {token_ids}\nexpert_ids: {expert_ids}\n")
            results[token_ids, :] += weights[token_ids, expert_ids].unsqueeze(1) * expert(x[token_ids]) # (N, 1) * (N, D) -> (N, D)

        return results


if __name__ == "__main__":
    B, S, D = 1, 3, 2
    E, K = 4, 2
    experts = [nn.Linear(D, D) for _ in range(E)]
    gate = nn.Linear(D, E)
    moe_layer = MoeLayer(experts, K, gate)
    x = torch.randn(B*S, D)
    results = moe_layer(x).reshape(B, S, D)
    print(f"results: {results}")
