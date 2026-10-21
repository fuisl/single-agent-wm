import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True, attn_mask=None):
        """
        x : (B, T, D); attn_mask (T, T) bool (True = attend) replaces the causal mask when given
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop,
                                             is_causal=causal and attn_mask is None)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None, attn_mask=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x, attn_mask=attn_mask) if isinstance(block, Block) else block(x, c, attn_mask=attn_mask)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


#################################################
## UAV additions: depth ViT, post-hoc readouts ##
#################################################

VIT_SIZES = {"tiny": (192, 12, 3), "small": (384, 12, 6)}  # hidden, layers, heads (as vit_hf)


def vit_depth(size="tiny", patch_size=8, image_size=(48, 80)):
    """HF ViT on 1-channel depth, configured like stable_pretraining's vit_hf(size, pretrained=False)."""
    from transformers import ViTConfig, ViTModel

    hidden, layers, heads = VIT_SIZES[size]
    config = ViTConfig(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
        intermediate_size=4 * hidden, image_size=list(image_size), patch_size=patch_size, num_channels=1,
    )
    return ViTModel(config, add_pooling_layer=False, use_mask_token=False)


class DepthDecoder(nn.Module):
    """Embedding -> 48x80 depth in [0, 1] (same scale as DreamerV3's decoder). Trained on frozen embeddings."""

    def __init__(self, input_dim, ch=64):
        super().__init__()
        self.ch = ch
        self.fc = nn.Sequential(nn.Linear(input_dim, 1024), nn.GELU(), nn.Linear(1024, 4 * ch * 3 * 5))
        up = lambda i, o: [nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.GELU()]
        self.net = nn.Sequential(
            *up(4 * ch, 4 * ch), *up(4 * ch, 2 * ch), *up(2 * ch, ch), *up(ch, ch),  # 3x5 -> 48x80
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, emb):
        """emb: (N, D) -> (N, 48, 80)"""
        x = self.fc(emb).view(-1, 4 * self.ch, 3, 5)
        return torch.sigmoid(self.net(x)[:, 0])


class PatchDecoder(nn.Module):
    """(D, 6, 10) patch-token grid -> 48x80 depth in [0, 1], the same up-sampling stack as DepthDecoder."""

    def __init__(self, input_dim, ch=64):
        super().__init__()
        up = lambda i, o: [nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.GELU()]
        self.net = nn.Sequential(nn.Conv2d(input_dim, 4 * ch, 1), *up(4 * ch, 2 * ch), *up(2 * ch, ch), *up(ch, ch),
                                 nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x):
        """x: (N, D, 6, 10) -> (N, 48, 80)"""
        return torch.sigmoid(self.net(x)[:, 0])


class RewardHead(nn.Module):
    """Embedding -> (reward, terminal logit) for the reward received on arriving at that observation."""

    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.net = MLP(input_dim, hidden_dim, 2)

    def forward(self, emb):
        out = self.net(emb)
        return out[..., 0], out[..., 1]


#################################################
## R-LeWM (docs/lewm_memory_review.md §5)      ##
#################################################

class TokenPredictor(nn.Module):
    """ARPredictor over several tokens per frame (M0 / M1): N pixel tokens + 1 proprio token per frame, frame-causal
    attention (a token sees every token of its own and earlier frames), AdaLN action conditioning per frame.
    The pixel-token outputs at frame t predict the pixel tokens of frame t+1, the proprio output the next state."""

    def __init__(self, *, num_frames, num_tokens, state_dim, depth, heads, mlp_dim, input_dim, hidden_dim,
                 dim_head=64, dropout=0.0):
        super().__init__()
        self.frame_pos = nn.Parameter(torch.randn(1, num_frames, 1, input_dim))
        self.token_pos = nn.Parameter(torch.randn(1, 1, num_tokens + 1, input_dim))
        self.state_in = nn.Linear(state_dim, input_dim)
        self.state_out = nn.Linear(input_dim, state_dim)
        self.transformer = Transformer(input_dim, hidden_dim, input_dim, depth, heads, dim_head, mlp_dim, dropout,
                                       block_class=ConditionalBlock)

    def forward(self, x, s, c):
        """x: (B, T, N, D) pixel tokens, s: (B, T, S) normalised state, c: (B, T, D) action embedding
        returns (B, T, N, D) pixel-token outputs, (B, T, S) next-state predictions"""
        B, T, N, D = x.shape
        x = torch.cat([x, self.state_in(s)[:, :, None]], 2) + self.frame_pos[:, :T] + self.token_pos
        frame = torch.arange(T, device=x.device).repeat_interleave(N + 1)
        mask = frame[:, None] >= frame[None, :]
        out = self.transformer(x.flatten(1, 2), c.repeat_interleave(N + 1, 1), attn_mask=mask)
        out = out.view(B, T, N + 1, D)
        return out[:, :, :N], self.state_out(out[:, :, N])


class Belief(nn.Module):
    """Recurrent belief over frame embeddings (M2 / M3), after CPC|Action / PBL / BYOL-Explore:
      closed loop  b_t = GRU(b_{t-1}, [z_t, s_t, a_{t-1}])     (z_t replaced by a learned blind vector when masked)
      open loop    o_0 = b_t, o_k = GRU'(o_{k-1}, a_{t+k-1})   (actions only)
      heads        o_k -> next pixel tokens z_{t+k}, next state s_{t+k}"""

    def __init__(self, num_tokens, token_dim, state_dim, action_dim, belief_dim=1024, input_dim=512):
        super().__init__()
        self.num_tokens, self.token_dim = num_tokens, token_dim
        self.obs_in = nn.Sequential(nn.Linear(num_tokens * token_dim + state_dim + action_dim, input_dim),
                                    nn.LayerNorm(input_dim), nn.SiLU())
        self.blind = nn.Parameter(torch.zeros(num_tokens * token_dim))
        self.closed = nn.GRUCell(input_dim, belief_dim)
        self.act_in = nn.Sequential(nn.Linear(action_dim, input_dim), nn.LayerNorm(input_dim), nn.SiLU())
        self.open = nn.GRUCell(input_dim, belief_dim)
        self.token_head = MLP(belief_dim, 2048, num_tokens * token_dim, norm_fn=nn.BatchNorm1d)
        self.state_head = MLP(belief_dim, 512, state_dim)
        self.belief_dim = belief_dim

    def observe(self, z, s, a_prev, blind=None, b0=None, reset=None):
        """z: (B, T, N, D), s: (B, T, S), a_prev: (B, T, A) action block that led to frame t (zeros at t=0),
        blind: (B, T) bool, reset: (B, T) bool (belief set to b0/zeros before step t). Returns (B, T, belief_dim)."""
        B, T = z.shape[:2]
        z = z.flatten(2)
        if blind is not None:
            z = torch.where(blind[..., None], self.blind.to(z.dtype).expand_as(z), z)
        x = self.obs_in(torch.cat([z, s, a_prev], -1).float())
        b = torch.zeros(B, self.belief_dim, device=z.device) if b0 is None else b0
        out = []
        for t in range(T):
            if reset is not None:
                b = torch.where(reset[:, t, None], torch.zeros_like(b), b)
            b = self.closed(x[:, t], b)
            out.append(b)
        return torch.stack(out, 1)

    def imagine(self, b, actions):
        """b: (M, belief_dim), actions: (M, K, A) -> pixel tokens (M, K, N, D), states (M, K, S), open states (M, K, Bd)"""
        o, outs = b, []
        for k in range(actions.size(1)):
            o = self.open(self.act_in(actions[:, k].float()), o)
            outs.append(o)
        o = torch.stack(outs, 1)
        M, K = o.shape[:2]
        tok = self.token_head(o.flatten(0, 1)).view(M, K, self.num_tokens, self.token_dim)
        return tok, self.state_head(o.flatten(0, 1)).view(M, K, -1), o


class MapBelief(Belief):
    """M4: the belief is an ego-centric latent map (channels x size x size cells of cell_m metres, drone at the centre
    facing up), transported by the drone's own motion and written by the current frame (in the spirit of Flow
    Equivariant World Models / EgoMap):
        M_t = (1 - g) * warp(M_{t-1}, ego-motion) + g * u,   [u, g] = lift(z_t, s_t)
        b_t = readout(M_t)  (a small CNN), then the open-loop head of Belief, unchanged.
    The ego-motion over one frame comes from the body state (velocity, yaw rate) at the previous frame, so nothing
    beyond the proprioception the other variants already get is used."""

    def __init__(self, num_tokens, token_dim, state_dim, action_dim, belief_dim=1024, input_dim=512,
                 channels=32, size=16, cell_m=0.6, dt=0.25, vel_idx=(5, 6), yaw_rate_idx=10):
        super().__init__(num_tokens, token_dim, state_dim, action_dim, belief_dim, input_dim)
        del self.closed
        self.channels, self.size, self.cell_m, self.dt = channels, size, cell_m, dt
        self.vel_idx, self.yaw_rate_idx = list(vel_idx), yaw_rate_idx
        self.lift = nn.Linear(input_dim, 2 * channels * size * size)
        self.readout = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Flatten(), nn.Linear(128 * (size // 4) ** 2, belief_dim))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

    def warp(self, m, s):
        """Move map content by the inverse ego-motion of one frame. m: (B, C, S, S), s: normalised state (B, D)."""
        raw = s.float() * self.state_std + self.state_mean
        v, w = raw[:, self.vel_idx] * self.dt, raw[:, self.yaw_rate_idx] * self.dt  # body-frame step, yaw change
        half = self.size * self.cell_m / 2
        c, sn = torch.cos(w), torch.sin(w)
        # map axes: image x = body left (-y), image y = body backward (-x); output cell p samples input at R p + t
        theta = torch.stack([torch.stack([c, sn, -v[:, 1] / half], -1),
                             torch.stack([-sn, c, -v[:, 0] / half], -1)], 1)
        grid = torch.nn.functional.affine_grid(theta, m.shape, align_corners=False)
        return torch.nn.functional.grid_sample(m, grid, align_corners=False)

    def observe(self, z, s, a_prev, blind=None, b0=None, reset=None, return_map=False):
        """As Belief.observe; b0 / reset act on the map. b0: (B, C, S, S) map or None. Returns beliefs (B, T, Bd)."""
        B, T = z.shape[:2]
        z = z.flatten(2)
        if blind is not None:
            z = torch.where(blind[..., None], self.blind.to(z.dtype).expand_as(z), z)
        x = self.obs_in(torch.cat([z, s, a_prev], -1).float())
        m = torch.zeros(B, self.channels, self.size, self.size, device=z.device) if b0 is None else b0
        out, maps = [], []
        for t in range(T):
            if reset is not None:
                m = torch.where(reset[:, t, None, None, None], torch.zeros_like(m), m)
            if t > 0:
                m = self.warp(m, s[:, t - 1])
            u, g = self.lift(x[:, t]).view(B, 2, self.channels, self.size, self.size).unbind(1)
            g = torch.sigmoid(g - 2.0)  # mostly keep the map at init
            m = (1 - g) * m + g * torch.tanh(u)
            out.append(self.readout(m))
            maps.append(m)
        return (torch.stack(out, 1), torch.stack(maps, 1)) if return_map else torch.stack(out, 1)
