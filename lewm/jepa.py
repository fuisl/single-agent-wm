"""JEPA Implementation (LeWM), adapted to depth + body state.

Changes from the official jepa.py:
  * encode: the 15-D state embedding is added to the ViT CLS token before the projector (the goal is only
    given through the state); no pos-encoding interpolation (fixed 48x80 input, non-square patch grid).
  * rollout works on embeddings (history encoded once per plan) and get_cost scores candidates with
    reward / terminal heads read out from the predicted embeddings, since this task has no goal image.
"""

import torch
from einops import rearrange
from torch import nn


class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        state_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.state_encoder = state_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels (B, T, 1, H, W), state (B, T, S) and optionally action (B, T, A) keys
        """

        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
        output = self.encoder(pixels)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        state_emb = rearrange(self.state_encoder(info["state"]), "b t d -> (b t) d")
        emb = self.projector(pixels_emb + state_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, emb, action, n_steps, history_size: int = 3):
        """Autoregressive rollout in embedding space.
        emb: (N, H, D) embeddings of the observed history (H <= history_size)
        action: (N, H - 1 + n_steps, A) action blocks: the H - 1 executed ones, then the n_steps to imagine
        returns (N, n_steps, D) predicted embeddings
        """
        H = emb.size(1)
        assert action.size(1) == H - 1 + n_steps, (emb.shape, action.shape, n_steps)
        act_emb = self.action_encoder(action)
        HS = history_size
        for t in range(n_steps):
            pred_emb = self.predict(emb[:, -HS:], act_emb[:, : H + t][:, -HS:])[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
        return emb[:, H:]

    def get_cost(self, heads, emb, action_hist, candidates, history_size: int = 3):
        """Negative expected return of action candidates.
        emb: (B, H, D) history embeddings, action_hist: (B, H - 1, A), candidates: (B, S, P, A)
        returns (B, S)
        """
        B, S, P = candidates.shape[:3]
        emb = emb.repeat_interleave(S, 0)
        action = torch.cat([action_hist.repeat_interleave(S, 0), rearrange(candidates, "b s p a -> (b s) p a")], 1)
        pred = self.rollout(emb, action, P, history_size)
        reward, term_logit = heads(pred)  # (B*S, P)
        alive = torch.cumprod(1 - torch.sigmoid(term_logit), 1)
        alive = torch.cat([torch.ones_like(alive[:, :1]), alive[:, :-1]], 1)  # reward of the step that terminates counts
        ret = (alive * reward).sum(1)
        return -rearrange(ret, "(b s) -> b s", b=B)
