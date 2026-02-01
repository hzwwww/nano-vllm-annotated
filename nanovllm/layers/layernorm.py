import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    y = x / (sqrt(sum(x^2)) + eps) * weights
    保持激活值在合理范围内：
        训练过程更加稳定
        避免梯度爆炸和消失
        避免过度依赖某一过大或过小的参数值，提高模型泛化能力
    """
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps # 保证数值计算稳定
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 不同的特征具备不同的重要性，增强模型表达能力

    @torch.compile # 优化：合并操作、合并内存访问，减少内存访问；优化不必要的操作
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True) # [bs, hidden_size] => [bs, 1]
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
