import torch


class CEMSolver:
    """Cross-entropy method over action-block sequences in normalised action space (cf. swm CEMSolver).

    Costs come from JEPA.get_cost: negative predicted return of the imagined embedding rollout.
    """

    def __init__(self, model, heads, num_samples=300, var_scale=1.0, n_steps=30, topk=30, seed=0,
                 history_size=3, device="cuda"):
        self.model, self.heads = model, heads
        self.num_samples, self.var_scale, self.n_steps, self.topk = num_samples, var_scale, n_steps, topk
        self.history_size = history_size
        self.gen = torch.Generator(device).manual_seed(seed)

    @torch.no_grad()
    def solve(self, emb, action_hist, horizon, low, high, init_mean=None):
        """emb (B, H, D), action_hist (B, H - 1, A), low/high (A,) bounds -> plan (B, horizon, A)"""
        B, A = emb.size(0), low.numel()
        mean = torch.zeros(B, horizon, A, device=emb.device) if init_mean is None else init_mean.clone()
        std = torch.full_like(mean, self.var_scale)
        for _ in range(self.n_steps):
            noise = torch.randn(B, self.num_samples, horizon, A, device=emb.device, generator=self.gen)
            cand = mean[:, None] + std[:, None] * noise
            cand[:, 0] = mean  # keep the current mean as a candidate
            cand = torch.maximum(torch.minimum(cand, high), low)
            cost = self.model.get_cost(self.heads, emb, action_hist, cand, self.history_size)
            elite = cand.gather(1, cost.topk(self.topk, 1, largest=False).indices[..., None, None]
                                .expand(-1, -1, horizon, A))
            mean, std = elite.mean(1), elite.std(1)
        return mean
