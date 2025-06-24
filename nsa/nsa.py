import torch
from typing import Optional
from torch.nn.attention.flex_attention import create_block_mask
from flash_attn import flash_attn_func

from fla.ops.utils.pooling import mean_pooling
from fla.ops.nsa.parallel import parallel_nsa_topk

from compression import compression_attention
from selection import selection_attention


def nsa_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: Optional[torch.Tensor] = None,
    g_slc: Optional[torch.Tensor] = None,
    g_swa: Optional[torch.Tensor] = None,
    block_count: int = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    B, M, H, D = q.shape
    _, N, G, _ = k.shape

    assert g_cmp is not None and g_slc is not None and g_swa is not None, "g_cmp, g_slc, and g_swa are required"
    assert k.shape == (B, N, G, D), f"k shape: {k.shape} must be ({B}, {N}, {G}, {D})"
    assert v.shape == (B, N, G, D), f"v shape: {v.shape} must be ({B}, {N}, {G}, {D})"
    assert g_cmp.shape == (B, M, H), f"g_cmp shape: {g_cmp.shape} must be ({B}, {M}, {H})"
    assert g_slc.shape == (B, M, H), f"g_slc shape: {g_slc.shape} must be ({B}, {M}, {H})"
    assert g_swa.shape == (B, M, H), f"g_swa shape: {g_swa.shape} must be ({B}, {M}, {H})"

    if scale is None:
        scale = D ** -0.5
    
    k_cmp, v_cmp = mean_pooling(k, block_size), mean_pooling(v, block_size)
    
    def cmp_mask(b, h, q_idx, kv_idx):
        return q_idx <= (kv_idx + 1) * block_size - 1
    
    block_mask = create_block_mask(cmp_mask, B, H, M, N//block_size)

    o_cmp, lse_cmp = compression_attention(q, k_cmp, v_cmp, block_mask)
    
    block_indices = parallel_nsa_topk(
        q=q,
        k=k_cmp,
        lse=lse_cmp,
        block_counts=block_count,
        block_size=block_size,
        scale=scale,
        cu_seqlens=None
    )
    
    o_slc = selection_attention(
        q, k, v, block_indices, block_count, block_size, scale
    )
    
    o_swd = flash_attn_func(
        q, k, v,
        causal=True,
        window_size=(window_size-1, 0)
    )

    o = o_cmp * g_cmp.unsqueeze(-1) + o_slc * g_slc.unsqueeze(-1) + o_swd * g_swa.unsqueeze(-1)
    return o