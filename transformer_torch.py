"""
Small causal transformer in PyTorch for the regime-sweep experiment.

1-layer, configurable num_heads and d_head. RoPE positional encoding.
Trained on next-symbol CE from ring-HMM generated data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RoPE(nn.Module):
    def __init__(self, d_head, max_len=2048):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, freqs)
        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())

    def forward(self, x):
        # x: (batch, seq, heads, d_head)
        T = x.size(1)
        d_half = x.size(-1) // 2
        cos = self.cos[:T, :d_half]  # (T, d_half)
        sin = self.sin[:T, :d_half]

        x1, x2 = x[..., :d_half], x[..., d_half:]
        out1 = x1 * cos[None, :, None, :] - x2 * sin[None, :, None, :]
        out2 = x2 * cos[None, :, None, :] + x1 * sin[None, :, None, :]
        return torch.cat([out1, out2], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, rope, dropout=0.0, residual_mode='additive'):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.rope = rope
        self.residual_mode = residual_mode

        self.ln1 = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        if residual_mode == 'gated':
            self.attn_gate = nn.Linear(d_model, d_model)
            self.ffn_gate = nn.Linear(d_model, d_model)
            nn.init.constant_(self.attn_gate.bias, 1.0)
            nn.init.constant_(self.ffn_gate.bias, 1.0)

    def _residual(self, x, delta, gate_layer=None):
        if self.residual_mode == 'additive':
            return x + delta
        elif self.residual_mode == 'gated':
            gate = torch.sigmoid(gate_layer(x))
            return gate * x + (1 - gate) * delta
        else:
            return delta

    def forward(self, x, mask, x_emb=None):
        B, T = x.shape[:2]
        H, D = self.num_heads, self.d_head

        x_ln = self.ln1(x)
        Q = self.W_q(x_ln).view(B, T, H, D)

        kv_source = self.ln1(x_emb) if x_emb is not None else x_ln
        K = self.W_k(kv_source).view(B, T, H, D)
        V = self.W_v(kv_source).view(B, T, H, D)
        Q = self.rope(Q)
        K = self.rope(K)

        scores = torch.einsum('bthd,bshd->bhts', Q, K) / math.sqrt(D)
        scores.masked_fill_(mask[None, None, :, :], float('-inf'))
        attn = self.attn_drop(F.softmax(scores, dim=-1))

        attn_out = torch.einsum('bhts,bshd->bthd', attn, V).reshape(B, T, self.d_model)
        attn_out = self.resid_drop(self.W_o(attn_out))
        x = self._residual(x, attn_out, getattr(self, 'attn_gate', None))
        x = self._residual(x, self.ffn(self.ln2(x)), getattr(self, 'ffn_gate', None))
        return x


class LinearAttentionBlock(nn.Module):
    """
    Causal linear attention (Katharopoulos et al. 2020).
    Normalized variant: output = phi(Q)^T S_t / (phi(Q)^T z_t).
    """

    def __init__(self, d_model, num_heads, rope, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.rope = rope

        self.ln1 = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.resid_drop = nn.Dropout(dropout)

    def _feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x, mask=None, x_emb=None):
        B, T = x.shape[:2]
        H, D = self.num_heads, self.d_head

        x_ln = self.ln1(x)
        Q = self.W_q(x_ln).view(B, T, H, D)
        K = self.W_k(x_ln).view(B, T, H, D)
        V = self.W_v(x_ln).view(B, T, H, D)

        Q = self.rope(Q)
        K = self.rope(K)

        Q = self._feature_map(Q)
        K = self._feature_map(K)

        KV = torch.einsum('bthd,bthe->bthde', K, V)
        KV_cumsum = torch.cumsum(KV, dim=1)
        K_cumsum = torch.cumsum(K, dim=1)

        numerator = torch.einsum('bthd,bthde->bthe', Q, KV_cumsum)
        denominator = torch.einsum('bthd,bthd->bth', Q, K_cumsum)
        denominator = denominator.unsqueeze(-1).clamp(min=1e-6)

        attn_out = (numerator / denominator).reshape(B, T, self.d_model)
        attn_out = self.resid_drop(self.W_o(attn_out))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class PhaseAttentionBlock(nn.Module):
    """
    Phase-space attention: unnormalized linear attention + periodic activation.

    Instead of normalizing by z_t (which reintroduces averaging), we let the
    cumulative state grow freely and apply sin/cos to map it to bounded [-1, 1].

    Why this works for belief filtering:
    - Cumulative state S_t = sum_{i<=t} phi(k_i) v_i^T grows ADDITIVELY
    - Each observation ADDS to the state (like log-likelihood accumulation)
    - sin/cos wrapping means overflow loops around instead of exploding
    - The PHASE relationships between dimensions encode regime identity
    - Multiple heads at different "frequencies" = Fourier-like decomposition
      of the belief state

    Connection to RoPE: RoPE encodes POSITION as rotation. This encodes
    accumulated EVIDENCE as rotation. Same math, different semantics.

    Connection to ring topology: regimes live on a ring. Phases live on a
    circle. Natural geometric match.
    """

    def __init__(self, d_model, num_heads, rope, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.rope = rope

        self.ln1 = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Learned frequency scaling per head — controls how fast phases rotate
        self.freq_scale = nn.Parameter(torch.ones(num_heads, 1) * 0.1)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.resid_drop = nn.Dropout(dropout)

    def _feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x, mask=None, x_emb=None):
        B, T = x.shape[:2]
        H, D = self.num_heads, self.d_head

        x_ln = self.ln1(x)
        Q = self.W_q(x_ln).view(B, T, H, D)
        K = self.W_k(x_ln).view(B, T, H, D)
        V = self.W_v(x_ln).view(B, T, H, D)

        Q = self.rope(Q)
        K = self.rope(K)

        Q = self._feature_map(Q)
        K = self._feature_map(K)

        # Cumulative state (no normalization)
        KV = torch.einsum('bthd,bthe->bthde', K, V)
        KV_cumsum = torch.cumsum(KV, dim=1)

        # Unnormalized output: phi(Q)^T S_t
        raw_out = torch.einsum('bthd,bthde->bthe', Q, KV_cumsum)  # (B, T, H, D)

        # Phase wrapping: scale by learned frequency, then sin/cos
        # Output both sin and cos to preserve full phase information
        # Each head's d_head dims split: first half gets sin, second half gets cos
        scaled = raw_out * self.freq_scale[None, None, :, :]  # (B, T, H, D)
        d_half = D // 2
        phase_out = torch.cat([
            torch.sin(scaled[..., :d_half]),
            torch.cos(scaled[..., d_half:]),
        ], dim=-1)  # (B, T, H, D)

        attn_out = phase_out.reshape(B, T, self.d_model)
        attn_out = self.resid_drop(self.W_o(attn_out))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, n_layers=1, max_len=2048,
                 dropout=0.0, reinject=False, attn_type='softmax',
                 residual_mode='additive', tie_weights=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.n_layers = n_layers
        self.reinject = reinject
        self.attn_type = attn_type
        self.residual_mode = residual_mode

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.rope = RoPE(self.d_head, max_len)

        block_map = {
            'softmax': TransformerBlock,
            'linear': LinearAttentionBlock,
            'phase': PhaseAttentionBlock,
        }
        BlockClass = block_map[attn_type]
        block_kwargs = dict(d_model=d_model, num_heads=num_heads,
                            rope=self.rope, dropout=dropout)
        if attn_type == 'softmax':
            block_kwargs['residual_mode'] = residual_mode
        self.blocks = nn.ModuleList([
            BlockClass(**block_kwargs)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        self._init_weights()

        # WEIGHT TYING. Without it, W_U[w] and emb(w) are unrelated
        # directions, so shaping a position's output distribution toward w
        # writes into a subspace that downstream attention -- whose keys
        # are built from emb -- cannot read. Tying makes "the distribution
        # here says heavy" identical to "emb(heavy) is in this residual",
        # which turns a DERIVED hop result into a literal the next lookup
        # can BIND against. Applied after _init_weights so it is not
        # overwritten.
        if tie_weights:
            self.output.weight = self.embedding.weight

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def _run_blocks(self, x, x_emb):
        B, T = x.shape[:2]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for i, block in enumerate(self.blocks):
            emb_for_kv = x_emb if (self.reinject and i > 0) else None
            x = block(x, mask, x_emb=emb_for_kv)
        return x

    def forward(self, tokens):
        B, T = tokens.shape
        x_emb = self.embedding(tokens)
        x = self.emb_drop(x_emb)
        x = self._run_blocks(x, x_emb)
        logits = self.output(self.ln_f(x))
        return logits

    def get_hidden(self, tokens):
        B, T = tokens.shape
        x_emb = self.embedding(tokens)
        x = self._run_blocks(x_emb, x_emb)
        return self.ln_f(x)
