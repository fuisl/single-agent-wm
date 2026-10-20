"""DreamerV3 agent extensions shared by offline (train_wm.py) and online (train_rl.py) training.

with_map_loss(base, scale, pos_weight) returns a subclass of `base` whose world model has an extra
head decoding the 64x64 ego-centric occupancy map from (h_t, z_t). The loss is a positive-weighted
BCE averaged over cells seen so far (`seen` mask); never-seen cells are not supervised. Gradients
flow into the RSSM. `occ` and `seen` are privileged simulator labels: they are kept out of the
encoder and decoder, so the policy never sees them.
"""
import embodied
import jax.numpy as jnp
import ninjax as nj
import optax
from embodied.jax import nets

MAP_KEYS = ("occ", "seen")


class MapHead(nj.Module):
    units: int = 1024

    def __init__(self):
        pass

    def __call__(self, x):
        x = self.sub("mlp", nets.MLP, 2, self.units)(x)
        return self.sub("out", nets.Linear, 64 * 64)(x)


def with_map_loss(base, scale, pos_weight=5.0):

    class MapAgent(base):

        def __init__(self, obs_space, act_space, config):
            enc_dec_space = {k: v for k, v in obs_space.items() if k not in MAP_KEYS}
            orig = embodied.jax.Optimizer
            holder = {}

            def make_opt(modules, opt, **kw):             # add the map head to the world-model optimizer
                holder["head"] = MapHead(name="maphead")
                return orig(list(modules) + [holder["head"]], opt, **kw)

            embodied.jax.Optimizer = make_opt
            try:
                super().__init__(enc_dec_space, act_space, config)
            finally:
                embodied.jax.Optimizer = orig
            self.maphead = holder["head"]
            self.obs_space = obs_space

        def loss(self, carry, obs, prevact, training):
            loss, (carry, entries, outs, metrics) = super().loss(carry, obs, prevact, training)
            B, T = obs["is_first"].shape
            logits = self.maphead(self.feat2tensor(outs["repfeat"])).astype(jnp.float32).reshape(B, T, -1)
            tgt = obs["occ"].reshape(B, T, -1).astype(jnp.float32)
            mask = obs["seen"].reshape(B, T, -1).astype(jnp.float32)
            bce = optax.sigmoid_binary_cross_entropy(logits, tgt) * (1 + (pos_weight - 1) * tgt)
            lmap = (bce * mask).sum(-1) / jnp.maximum(mask.sum(-1), 1.0)
            metrics["loss/map"] = lmap.mean()
            return loss + scale * lmap.mean(), (carry, entries, outs, metrics)

    MapAgent.__name__ = f"MapAgent_{base.__name__}"
    return MapAgent
