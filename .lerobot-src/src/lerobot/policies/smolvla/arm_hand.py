"""Arm/hand feature branches for flow matching; both limbs share the action expert."""

import torch
from torch import Tensor, nn


def masked_action_mean(loss: Tensor, is_pad: Tensor | None) -> Tensor:
    """Per-example mean over real timesteps and the supplied (unpadded) joints."""
    if is_pad is None:
        return loss.mean(dim=(1, 2))
    valid = (~is_pad).unsqueeze(-1)
    denominator = (valid.sum(dim=(1, 2)) * loss.shape[-1]).clamp_min(1)
    return (loss * valid).sum(dim=(1, 2)) / denominator


class ArmHandFeatures(nn.Module):
    """Dedicated MLP features and auxiliary velocity heads (paper Sec. 3.4).

    The main output is W_shared*z + W_limbs*[z_arm,z_hand] + b.
    Keeping the existing shared projection and zeroing W_limbs preserves a
    pretrained policy's predictions at initialization, while auxiliary heads
    immediately provide gradients to the new feature branches.
    """

    def __init__(self, shared_dim: int, hidden_dim: int, action_dim: int, arm_dim: int, hand_dim: int):
        super().__init__()
        self.arm_encoder = nn.Sequential(nn.Linear(shared_dim, hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.hand_encoder = nn.Sequential(nn.Linear(shared_dim, hidden_dim), nn.SiLU(),
                                          nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.arm_head = nn.Linear(hidden_dim, arm_dim)
        self.hand_head = nn.Linear(hidden_dim, hand_dim)
        self.fusion = nn.Linear(2 * hidden_dim, action_dim, bias=False)
        nn.init.zeros_(self.fusion.weight)

    def forward(self, shared: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        arm = self.arm_encoder(shared)
        hand = self.hand_encoder(shared)
        return self.fusion(torch.cat((arm, hand), dim=-1)), self.arm_head(arm), self.hand_head(hand)
