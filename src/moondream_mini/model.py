from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MiniConfig:
    vocab_size: int = 32000
    image_size: int = 160
    patch_size: int = 16
    num_image_tokens: int = 101  # (160 / 16)^2 + cls
    vision_dim: int = 128
    text_dim: int = 128
    num_heads: int = 4
    num_vision_layers: int = 2
    num_text_layers: int = 3
    ff_dim: int = 384
    max_text_len: int = 128
    dropout: float = 0.1
    # The final v6 checkpoint was trained with causal attention inside the
    # vision blocks. Keep that behavior as the compatibility default.
    vision_is_causal: bool = True


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        is_causal: bool = False,
    ):
        b, t, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.dropout(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        is_causal: bool = False,
    ):
        x = x + self.attn(self.ln1(x), attn_mask, is_causal=is_causal)
        x = x + self.mlp(self.ln2(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, cfg: MiniConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(3, cfg.vision_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.base_grid = cfg.image_size // cfg.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.vision_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.vision_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.vision_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout)
                for _ in range(cfg.num_vision_layers)
            ]
        )
        self.ln = nn.LayerNorm(cfg.vision_dim)
        self.proj = nn.Linear(cfg.vision_dim, cfg.text_dim)

    def _positional_encoding(self, h: int, w: int, device, dtype):
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        if h * w != self.base_grid * self.base_grid:
            patch_pos = patch_pos.reshape(1, self.base_grid, self.base_grid, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bilinear", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return torch.cat([cls_pos, patch_pos], dim=1).to(device=device, dtype=dtype)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)  # (B, D, H, W)
        h, w = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(images.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._positional_encoding(h, w, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, is_causal=self.cfg.vision_is_causal)
        x = self.ln(x)
        return self.proj(x)


class MiniMoondream(nn.Module):
    def __init__(self, cfg: MiniConfig):
        super().__init__()
        self.cfg = cfg
        self.vision = VisionEncoder(cfg)
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.text_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_text_len + cfg.num_image_tokens, cfg.text_dim))
        self.vision_token_type = nn.Parameter(torch.zeros(1, 1, cfg.text_dim))
        self.text_token_type = nn.Parameter(torch.zeros(1, 1, cfg.text_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.text_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout) for _ in range(cfg.num_text_layers)]
        )
        self.ln = nn.LayerNorm(cfg.text_dim)
        self.lm_head = nn.Linear(cfg.text_dim, cfg.vocab_size, bias=False)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision(images)

    def build_inputs(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        img_tokens = self.encode_image(images)
        txt_tokens = self.token_embed(input_ids)
        img_tokens = img_tokens + self.vision_token_type
        txt_tokens = txt_tokens + self.text_token_type
        x = torch.cat([img_tokens, txt_tokens], dim=1)
        pos = self.pos_embed[:, : x.size(1)]
        return x + pos

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.build_inputs(images, input_ids)
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        if attention_mask is not None:
            key_mask = ~attention_mask.bool()
            prefix = torch.zeros((attention_mask.size(0), self.cfg.num_image_tokens), device=x.device, dtype=torch.bool)
            key_mask = torch.cat([prefix, key_mask], dim=1)
            key_mask = key_mask[:, None, None, :].expand(-1, 1, seq_len, -1)
            attn_mask = causal_mask[None, None, :, :].expand(x.size(0), 1, -1, -1) & ~key_mask
        else:
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.ln(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        repetition_penalty: float = 1.08,
        no_repeat_ngram_size: int = 3,
    ):
        self.eval()
        device = next(self.parameters()).device
        prompt_ids = tokenizer.encode(prompt).ids
        ids = torch.tensor([prompt_ids], device=device)
        image = image.to(device)
        eos_id = tokenizer.token_to_id("</s>")
        if eos_id is None:
            eos_id = tokenizer.token_to_id("<eos>")
        newline_id = tokenizer.token_to_id("\n")

        def has_repeated_ngram(seq, n):
            if n <= 0 or len(seq) < 2 * n:
                return False
            seen = set()
            for i in range(len(seq) - n + 1):
                ng = tuple(seq[i : i + n])
                if ng in seen:
                    return True
                seen.add(ng)
            return False

        for _ in range(max_new_tokens):
            logits = self.forward(image, ids)
            next_logits = logits[:, -1, :]

            if repetition_penalty and repetition_penalty != 1.0:
                for token_id in set(ids[0].tolist()[-32:]):
                    if next_logits[0, token_id] < 0:
                        next_logits[0, token_id] *= repetition_penalty
                    else:
                        next_logits[0, token_id] /= repetition_penalty

            if temperature <= 0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / max(temperature, 1e-6), dim=-1)
                next_id = torch.multinomial(probs, 1)

            ids = torch.cat([ids, next_id], dim=1)
            seq = ids[0].tolist()
            if eos_id is not None and next_id.item() == eos_id:
                break
            if newline_id is not None and next_id.item() == newline_id:
                break
            if no_repeat_ngram_size > 0 and has_repeated_ngram(seq[len(prompt_ids) :], no_repeat_ngram_size):
                break

        return tokenizer.decode(ids[0].tolist())
