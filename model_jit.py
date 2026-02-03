# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, RotaryEmbedding1D, get_2d_sincos_pos_embed, RMSNorm, get_1d_sincos_pos_embed_from_grid
import numpy as np


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class VideoPatchEmbed(nn.Module):
    """ Video to Tubelet Embedding
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=6, embed_dim=768, bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # Input: (B, C, T, H, W)
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                              kernel_size=(1, patch_size, patch_size), 
                              stride=(1, patch_size, patch_size), 
                              bias=bias)
        
        self.num_patches_spatial = (img_size // patch_size) ** 2

    def forward(self, x):
        # print(x.shape)
        # x: B, C, T, H, W
        x = self.proj(x) # 输出: B, D, T, H/p, W/p (5D)
        # print(x.shape)
        
        # 必须 flatten 空间维度 (H/p, W/p) -> L
        x = x.flatten(3) # 输出: B, D, T, L (4D)
        # print(x.shape)
        
        # 必须把 Channel 维度 (D) 移到最后，以匹配 Transformer 输入
        x = x.permute(0, 2, 3, 1) # 输出: B, T, L, D (4D)
        # print(x.shape)
        
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype, device=query.device)

    with torch.amp.autocast(device_type="cuda", enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # c: B, D -> B, 1, 1, D to modulate
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn_s = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.attn_t = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                attn_drop=attn_drop, proj_drop=proj_drop)

        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        
        # AdaLN modulation: 9 chunks
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, rope_s, rope_t):
        if x.ndim != 4:
            print(f"RANK DEBUG: x shape is {x.shape} (Expected 4 dims)")
        # x: B, T, L, D. 如果这里报错，说明 VideoPatchEmbed 输出维度不对
        # print("before jitblock", x.shape)
        B, T, L, D = x.shape
        
        chunks = self.adaLN_modulation(c).chunk(9, dim=-1)
        shift_msa_s, scale_msa_s, gate_msa_s = chunks[0], chunks[1], chunks[2]
        shift_msa_t, scale_msa_t, gate_msa_t = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # 1. Spatial Attention
        x_s = x.reshape(B * T, L, D)
        
        shift_msa_s = shift_msa_s.unsqueeze(1).repeat(1, T, 1).view(B*T, D)
        scale_msa_s = scale_msa_s.unsqueeze(1).repeat(1, T, 1).view(B*T, D)
        gate_msa_s = gate_msa_s.unsqueeze(1).repeat(1, T, 1).view(B*T, D)

        x_s_norm = modulate(self.norm1(x_s), shift_msa_s, scale_msa_s)
        x_s = x_s + gate_msa_s.unsqueeze(1) * self.attn_s(x_s_norm, rope=rope_s)
        x = x_s.view(B, T, L, D)

        # 2. Temporal Attention
        x_t = x.permute(0, 2, 1, 3).reshape(B * L, T, D)

        shift_msa_t = shift_msa_t.unsqueeze(1).repeat(1, L, 1).view(B*L, D)
        scale_msa_t = scale_msa_t.unsqueeze(1).repeat(1, L, 1).view(B*L, D)
        gate_msa_t = gate_msa_t.unsqueeze(1).repeat(1, L, 1).view(B*L, D)

        x_t_norm = modulate(self.norm2(x_t), shift_msa_t, scale_msa_t)
        x_t = x_t + gate_msa_t.unsqueeze(1) * self.attn_t(x_t_norm, rope=rope_t)
        x = x_t.view(B, L, T, D).permute(0, 2, 1, 3) 
        # print("after temp attn", x.shape)

        # 3. MLP
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1).unsqueeze(1)

        # print("before mlp", x.shape)
        # print("shift_mlp", shift_mlp.shape)
        # print("scale_mlp", scale_mlp.shape)
        # print("gate_mlp", gate_mlp.shape)
        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        
        # print("after jitblock", x.shape)
        return x


class JiT(nn.Module):
    """
    Just image Transformer (Modified for Video I2V)
    """
    def __init__(
        self,
        input_size=128,
        patch_size=16,
        in_channels=6, 
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_frames=16,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 3 
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_frames = num_frames

        # time embed ONLY
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Patch Embed
        self.x_embedder = VideoPatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        # Positional Embeddings
        num_patches_spatial = self.x_embedder.num_patches_spatial
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 1, num_patches_spatial, hidden_size), requires_grad=False)
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, num_frames, 1, hidden_size), requires_grad=False)

        # RoPE
        head_dim = hidden_size // num_heads # FIX: Use full head dimension for calc
        half_head_dim = head_dim // 2
        hw_seq_len = input_size // patch_size
        
        self.feat_rope_s = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        
        # FIX: Temporal RoPE uses full head_dim
        self.feat_rope_t = RotaryEmbedding1D(
            dim=head_dim, 
            max_seq_len=num_frames
        )

        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop, proj_drop=proj_drop)
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed_s = get_2d_sincos_pos_embed(self.hidden_size, int(self.x_embedder.num_patches_spatial ** 0.5))
        self.pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0).unsqueeze(0))

        pos_embed_t = get_1d_sincos_pos_embed_from_grid(self.hidden_size, np.arange(self.num_frames))
        self.pos_embed_temporal.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0).unsqueeze(2))

        nn.init.xavier_uniform_(self.x_embedder.proj.weight)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        c = self.out_channels
        h = w = int(x.shape[2] ** 0.5)
        # print("before unpatchify", x.shape)
        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p, p, c))
        x = torch.einsum('nthwpqc->ncthpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, x.shape[2], h * p, w * p))
        return imgs

    def forward(self, x, t): 
        # x: (B, C_in, T, H, W)
        t_emb = self.t_embedder(t)
        c = t_emb 

        x = self.x_embedder(x) # -> B, T, L, D (4D)
        # print("after emb", x.shape)
        
        # Broadcasting Add
        x = x + self.pos_embed_spatial + self.pos_embed_temporal
        # print("after add", x.shape)

        for block in self.blocks:
            x = block(x, c, self.feat_rope_s, self.feat_rope_t)

        x = self.final_layer(x, c)
        # print("after final layer", x.shape)
        output = self.unpatchify(x, self.patch_size)

        return output


def JiT_B_16_I2V(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12, patch_size=16, **kwargs)

def JiT_L_16_I2V(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16, patch_size=16, **kwargs)

def JiT_H_16_I2V(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16, patch_size=16, **kwargs)

JiT_models = {
    'JiT-B/16': JiT_B_16_I2V,
    'JiT-L/16': JiT_L_16_I2V,
    'JiT-H/16': JiT_H_16_I2V,
}