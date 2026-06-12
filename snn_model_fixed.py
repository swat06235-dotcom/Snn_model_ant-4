import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
import os
from torch.optim import Adam

# ==============================================================================
#  변경 이력 (사실 기반 수정 — 2026-06-11)
# ------------------------------------------------------------------------------
#  [증명된 수정 — 실험으로 검증됨]
#   • surrogate_alpha = π  (기존 10.0 폐기).
#     이유: arctan surrogate 최대 기울기 = α/π. α=10이면 3.18 → 자기재귀 SNN에서
#     시간축 곱셈으로 그래디언트 폭발(probe서 ‖g‖=∞, NaN). α=π면 최대기울기=1.0
#     → 자코비안 ~1로 묶임(probe R²0.88, ‖g‖1.0, 3시드 재현). 폭발 소멸.
#   • reset 그래디언트 유지(detach 금지) — 이미 코드가 그렇게 돼 있어 그대로 둠.
#     이유: reset 항 -z·v_th의 음의 그래디언트가 자코비안을 깎는 '브레이크'.
#     detach하면 폭발(2×2 ablation 확정). → 절대 z.detach() 넣지 말 것.
#
#  [버그 수정 — 원본은 실행 불가였음]
#   • SACAPEXAgent가 없는 클래스 APEXActorSNN을 호출 → ActorSNN_Temporal로 수정.
#   • ActorSNN_Temporal이 ALIFCell(hidden, hidden, surr)로 호출 → surr가 T 자리에
#     들어가던 버그. ALIFCell(hidden, hidden, T=T, surrogate_alpha=surr)로 수정.
#   • ALIFCell의 self.rho / self.bias 중복 정의 제거.
#   • 떠 있던 클래스 docstring을 클래스 안으로 이동.
#
#  [건드리지 않은 것 — 증명되지 않은 설계는 보존]
#   • 항상성(Pop-Art/아드레날린/멜라토닌/SHY), DroQ, CoT, T-불변 감쇠 등 그대로.
#   ※ 주의(APEX_PHILOSOPHY.md 섹션 12): 완전관측 Ant에서 SNN의 '시간'은 보상을
#     올리지 못함(실험 확인). 이 코드는 그 한계를 안고 도는 baseline이다.
# ==============================================================================

# ─────────────────────────────────────────────
# 1. SURROGATE GRADIENT OPERATOR
# ─────────────────────────────────────────────

class SurrGradSpike(torch.autograd.Function):
    """
    Arctangent-based Surrogate Gradient for non-differentiable spiking.
    Forward:  Heaviside step function H(x)
    Backward: α / (π · (1 + (αx)²))   — 최대 기울기 = α/π

    α=π 로 두면 최대 기울기 = 1.0 → 자기재귀 SNN 자코비안 안정(폭발 방지).
    """
    @staticmethod
    def forward(ctx, input, alpha=math.pi):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / (math.pi * (1.0 + (alpha * input).pow(2)))
        return grad_output * grad, None


def surrogate_spike(x, alpha=math.pi):
    return SurrGradSpike.apply(x, alpha)


# ─────────────────────────────────────────────
# 2. ADAPTIVE LIF CELL
# ─────────────────────────────────────────────

class ALIFCell(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire neuron cell.

    State tuple: (v, a, z)
      v  – membrane potential   (batch, out_features)
      a  – adaptation variable  (batch, out_features)
      z  – previous spike       (batch, out_features)

    The recurrent weight W maps z_{t-1} → current input so the cell
    is self-recurrent.  The caller feeds feedforward current separately.

    ⚠ reset 항 (-z·v_th) 의 그래디언트는 끊지 말 것(detach 금지).
      그 음의 그래디언트가 자코비안을 안정화하는 내장 브레이크다.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        T: int,                       # T 스텝 후 보존율 → 스텝당 감쇠율 역산에 사용
        v_retention: float = 0.327,   # 제1원칙: T 스텝 후 막전위 물리적 보존율
        a_retention: float = 0.590,   # 제1원칙: T 스텝 후 적응변수 물리적 보존율
        v_th_0: float = 1.0,
        rho_init: float = 0.5,
        surrogate_alpha: float = math.pi,   # ★ 증명된 값: π (기존 10.0 폐기)
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.v_th_0 = v_th_0
        self.surrogate_alpha = surrogate_alpha

        self.ff_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.ff_weight, a=math.sqrt(5))
        self.rec_weight = nn.Parameter(torch.empty(out_features, out_features))
        nn.init.orthogonal_(self.rec_weight, gain=0.5)

        # ─── [핵심 논리] T 값에 따른 자동 감쇠율 계산 (제1법칙 적용) ───
        # decay = sigmoid(raw) 가 retention^(1/T) 가 되도록 logit 공간 초기화.
        beta_init = v_retention ** (1.0 / T)
        alpha_init = a_retention ** (1.0 / T)
        self.beta_raw = nn.Parameter(
            torch.full((out_features,), math.log(beta_init / (1.0 - beta_init)))
        )
        self.alpha_raw = nn.Parameter(
            torch.full((out_features,), math.log(alpha_init / (1.0 - alpha_init)))
        )

        # Threshold coupling constant (learnable) — 단일 정의
        self.rho = nn.Parameter(torch.full((out_features,), rho_init))
        # Bias for feed-forward current — 단일 정의
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x_in, state):
        """
        x_in : (batch, in_features)  – feed-forward pre-synaptic input
        state : tuple (v, a, z) each (batch, out_features)
        Returns z_next ∈ {0,1}, (v_next, a_next, z_next)
        """
        v, a, z = state

        beta = torch.sigmoid(self.beta_raw)
        alpha = torch.sigmoid(self.alpha_raw)

        # Adaptive threshold: v_th = v_th_0 + ρ · a
        v_th = self.v_th_0 + self.rho * a

        # Input current = W_ff · x_in + W_rec · z_{τ-1} + bias
        I = (
            F.linear(x_in, self.ff_weight, self.bias)
            + F.linear(z, self.rec_weight)
        )

        # Membrane update:  v[τ] = β·v[τ-1] + (1-β)·I - z[τ-1]·v_th  (reset 그래디언트 유지)
        v_next = beta * v + (1.0 - beta) * I - z * v_th

        # Spike (forward: 진짜 0/1, backward: surrogate α=π)
        z_next = surrogate_spike(v_next - v_th, self.surrogate_alpha)

        # Adaptation update: a[τ] = α·a[τ-1] + (1-α)·z[τ]
        a_next = alpha * a + (1.0 - alpha) * z_next

        return z_next, (v_next, a_next, z_next)

    def init_state(self, batch_size: int, device: torch.device):
        z = torch.zeros(batch_size, self.out_features, device=device)
        return (z.clone(), z.clone(), z.clone())  # (v, a, z)


# ─────────────────────────────────────────────
# 3. APEX SNN ACTOR
# ─────────────────────────────────────────────

class ActorSNN_Temporal(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(self, obs_dim, act_dim, hidden=128, T=8, surr=math.pi):
        super().__init__()
        self.T = T
        self.hidden = hidden

        self.input_proj = nn.Linear(obs_dim, hidden)
        # ★ 버그 수정: T와 surrogate_alpha를 올바른 인자로 전달
        self.alif1 = ALIFCell(hidden, hidden, T=T, surrogate_alpha=surr)
        self.alif2 = ALIFCell(hidden, hidden, T=T, surrogate_alpha=surr)

        self.mean_lin = nn.Linear(hidden, act_dim)
        self.logstd_lin = nn.Linear(hidden, act_dim)
        for lin in (self.mean_lin, self.logstd_lin):
            nn.init.uniform_(lin.weight, -3e-3, 3e-3)
            nn.init.uniform_(lin.bias,  -3e-3, 3e-3)

        # 시간 가중치(버퍼 → 디바이스 자동 동기화)
        times = torch.arange(T, dtype=torch.float32)
        # Latency coding: 입력 펄스 초기 강함
        in_burst = torch.exp(-times / (T / 2.0))
        self.register_buffer("in_burst", in_burst.view(T, 1, 1))
        # PSP 지수 감쇠 readout: 최근 스파이크 가중
        out_decay = torch.exp(-times / (T / 3.0))
        out_decay = out_decay / out_decay.sum()
        self.register_buffer("out_decay", out_decay.view(T, 1, 1))

    def forward(self, obs):
        B = obs.shape[0]
        dev = obs.device
        base_cur = self.input_proj(obs)             # [B, Hidden]

        s1 = self.alif1.init_state(B, dev)
        s2 = self.alif2.init_state(B, dev)

        z_history = []
        for t in range(self.T):
            current_in = base_cur * self.in_burst[t].squeeze(-1)  # latency coding
            z1, s1 = self.alif1(current_in, s1)
            z2, s2 = self.alif2(z1, s2)
            z_history.append(z2)

        z_stack = torch.stack(z_history, dim=0)     # [T, B, Hidden]
        rep = (z_stack * self.out_decay).sum(dim=0) # temporal weighted readout

        mean = self.mean_lin(rep)
        logstd = torch.clamp(self.logstd_lin(rep), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, logstd

    def sample(self, obs, eps=1e-6):
        mean, logstd = self.forward(obs)
        std = logstd.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        logp = normal.log_prob(x) - torch.log(1.0 - action.pow(2) + eps)
        logp = logp.sum(1, keepdim=True)
        return action, logp, torch.tanh(mean)


# ─────────────────────────────────────────────
# 4. DOUBLE-Q CRITIC (Continuous MLP)
# ─────────────────────────────────────────────

def _droq_body(in_dim: int, hidden_dim: int, depth: int, dropout: float = 0.01):
    """DroQ-style MLP body: Linear → Dropout → LayerNorm → ReLU."""
    layers = []
    d = in_dim
    for _ in range(depth):
        layers += [
            nn.Linear(d, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        d = hidden_dim
    return nn.Sequential(*layers)


class DoubleQCritic(nn.Module):
    """Double-Q critic with DroQ stabilisation + Pop-Art target normalisation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_depth: int = 2,
        dropout: float = 0.01,
        popart_beta: float = 1e-3,
    ):
        super().__init__()
        in_dim = obs_dim + action_dim

        self.body1 = _droq_body(in_dim, hidden_dim, hidden_depth, dropout)
        self.body2 = _droq_body(in_dim, hidden_dim, hidden_depth, dropout)
        self.head1 = nn.Linear(hidden_dim, 1)
        self.head2 = nn.Linear(hidden_dim, 1)

        self.popart_beta = popart_beta
        self.register_buffer("mu",            torch.zeros(1))
        self.register_buffer("sigma",         torch.ones(1))
        self.register_buffer("second_moment", torch.ones(1))

        self.apply(self._init_weights)
        for head in (self.head1, self.head2):
            nn.init.uniform_(head.weight, -3e-3, 3e-3)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)

    def normalized(self, obs: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([obs, action], dim=-1)
        return self.head1(self.body1(sa)), self.head2(self.body2(sa))

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        n1, n2 = self.normalized(obs, action)
        return self.mu + self.sigma * n1, self.mu + self.sigma * n2

    def q_min(self, obs: torch.Tensor, action: torch.Tensor):
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)

    @torch.no_grad()
    def update_popart(self, target: torch.Tensor):
        old_mu, old_sigma = self.mu.clone(), self.sigma.clone()
        beta = self.popart_beta
        self.mu.mul_(1 - beta).add_(beta * target.mean())
        self.second_moment.mul_(1 - beta).add_(beta * (target * target).mean())
        self.sigma.copy_((self.second_moment - self.mu * self.mu).clamp_min(1e-4).sqrt())

        scale = old_sigma / self.sigma
        shift = (old_mu - self.mu) / self.sigma
        self.apply_popart_rescale(scale, shift)
        return scale, shift

    @torch.no_grad()
    def apply_popart_rescale(self, scale, shift):
        for head in (self.head1, self.head2):
            head.weight.mul_(scale)
            head.bias.mul_(scale).add_(shift)

    @torch.no_grad()
    def copy_popart_stats_from(self, other):
        self.mu.copy_(other.mu)
        self.sigma.copy_(other.sigma)
        self.second_moment.copy_(other.second_moment)


# ─────────────────────────────────────────────
# 5. REPLAY BUFFER
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.action   = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward   = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.done     = np.zeros((capacity, 1),          dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.action[self.ptr]   = action
        self.reward[self.ptr]   = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr]     = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx],      device=device),
            torch.as_tensor(self.action[idx],   device=device),
            torch.as_tensor(self.reward[idx],   device=device),
            torch.as_tensor(self.next_obs[idx], device=device),
            torch.as_tensor(self.done[idx],     device=device),
        )

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────
# 6. SAC-APEX AGENT
# ─────────────────────────────────────────────

class SACAPEXAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        T: int = 5,
        surrogate_alpha: float = math.pi,    # ★ 증명된 값
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 1.0,
        divergence_threshold: float = 10.0,
        target_entropy: float = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.divergence_threshold = divergence_threshold
        self.action_dim = action_dim

        # ── Networks ──────────────────────────────────────────────────────
        # ★ 버그 수정: APEXActorSNN → ActorSNN_Temporal, surr=π 명시 전달
        self.actor = ActorSNN_Temporal(
            obs_dim, action_dim, hidden=hidden_dim, T=T, surr=surrogate_alpha
        ).to(self.device)

        self.critic        = DoubleQCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Entropy Temperature ───────────────────────────────────────────
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(
            math.log(0.2), dtype=torch.float32, requires_grad=True, device=self.device
        )

        # ── Optimisers ────────────────────────────────────────────────────
        self.actor_opt  = Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr_critic)
        self.alpha_opt  = Adam([self.log_alpha],         lr=lr_alpha)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            _, _, mean = self.actor.sample(obs_t)
            action = mean
        else:
            action, _, _ = self.actor.sample(obs_t)
        return action.cpu().numpy().squeeze(0)

    def _update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            q_target = torch.min(q1_t, q2_t) - self.alpha * next_log_prob
            y = self.reward_scale * reward + self.gamma * (1.0 - done) * q_target

        # [세로토닌] Pop-Art 항상성 갱신 (타깃망도 동일 프레임으로 — 불변식)
        scale, shift = self.critic.update_popart(y)
        self.critic_target.apply_popart_rescale(scale, shift)
        self.critic_target.copy_popart_stats_from(self.critic)
        y_norm = (y - self.critic.mu) / self.critic.sigma

        n1, n2 = self.critic.normalized(obs, action)

        with torch.no_grad():
            diverged = max(n1.abs().max().item(), n2.abs().max().item()) > self.divergence_threshold

        critic_loss = F.smooth_l1_loss(n1, y_norm) + F.smooth_l1_loss(n2, y_norm)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        max_norm = 0.5 if diverged else 2.0
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=max_norm)
        self.critic_opt.step()
        return critic_loss.item(), diverged

    def _update_actor(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        q_min = self.critic.q_min(obs, action)
        actor_loss = (self.alpha.detach() * log_prob - q_min).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
        self.actor_opt.step()
        return actor_loss.item(), log_prob.mean().item()

    def _update_alpha(self, log_prob: float):
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        return alpha_loss.item()

    def _soft_update_target(self):
        with torch.no_grad():
            for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_tgt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

    def update(self, replay_buffer, batch_size=256, utd=1, weight_decay=0.0, update_actor=True):
        critic_loss = 0.0
        diverged = False
        for _ in range(utd):
            obs, action, reward, next_obs, done = replay_buffer.sample(batch_size, self.device)
            critic_loss, d = self._update_critic(obs, action, reward, next_obs, done)
            diverged = diverged or d
            self._soft_update_target()

        actor_loss = 0.0
        mean_log_prob = 0.0
        alpha_loss = 0.0
        if update_actor:
            obs, *_ = replay_buffer.sample(batch_size, self.device)
            actor_loss, mean_log_prob = self._update_actor(obs)
            alpha_loss = self._update_alpha(torch.tensor(mean_log_prob, device=self.device))

        if weight_decay > 0.0:
            with torch.no_grad():
                for p in self.critic.parameters():
                    p.mul_(1.0 - weight_decay)

        return {
            "critic_loss": critic_loss,
            "actor_loss":  actor_loss,
            "alpha_loss":  alpha_loss,
            "alpha":       self.alpha.item(),
            "diverged":    diverged,
            "q_scale":     self.critic.sigma.item(),
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),         os.path.join(path, "actor.pt"))
        torch.save(self.critic.state_dict(),        os.path.join(path, "critic.pt"))
        torch.save(self.critic_target.state_dict(), os.path.join(path, "critic_target.pt"))
        torch.save(self.log_alpha,                  os.path.join(path, "log_alpha.pt"))

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pt"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pt"), map_location=self.device))
        self.critic_target.load_state_dict(torch.load(os.path.join(path, "critic_target.pt"), map_location=self.device))
        self.log_alpha.data = torch.load(os.path.join(path, "log_alpha.pt"), map_location=self.device)


# ─────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────

def compute_cot(torques: list, forward_displacements: list) -> float:
    total_torque_sq = sum(np.sum(u ** 2) for u in torques)
    total_displacement = sum(d for d in forward_displacements if d > 0)
    if total_displacement < 1e-8:
        return float("inf")
    return total_torque_sq / total_displacement


def train(
    env_id: str = "Ant-v4",
    total_steps: int = 1_000_000,
    batch_size: int = 256,
    replay_capacity: int = 1_000_000,
    warmup_steps: int = 10_000,
    eval_interval: int = 10_000,
    eval_episodes: int = 5,
    hidden_dim: int = 256,
    T: int = 5,
    surrogate_alpha: float = math.pi,        # ★ 증명된 값
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    lr_alpha: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    reward_scale: float = 1.0,
    seed: int = 42,
    device: str = "cuda",
    save_dir: str = "./apex_checkpoints",
    log_interval: int = 1000,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env      = gym.make(env_id)
    eval_env = gym.make(env_id)
    env.action_space.seed(seed)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"[SNN-APEX] env={env_id}  obs={obs_dim}  act={action_dim}  "
          f"device={device}  T={T}  surr_alpha={surrogate_alpha:.4f}  seed={seed}")

    agent = SACAPEXAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        T=T,
        surrogate_alpha=surrogate_alpha,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        lr_alpha=lr_alpha,
        gamma=gamma,
        tau=tau,
        reward_scale=reward_scale,
        target_entropy=-float(action_dim),
        device=device,
    )
    buffer = ReplayBuffer(obs_dim, action_dim, capacity=replay_capacity)

    step = 0
    episode = 0
    ep_reward = 0.0
    ep_torques: list = []
    ep_displacements: list = []
    best_eval_reward = -np.inf

    obs, _ = env.reset(seed=seed)

    # ── [제1원칙] 항상성·리듬·비상 분리 (세로토닌/멜라토닌/아드레날린) ──
    WAKE_UTD           = 4
    SLEEP_INTERVAL     = 5000
    SLEEP_UPDATES      = 250
    SLEEP_WEIGHT_DECAY = 1e-5
    ALPHA_LR_FLOOR     = 1e-5

    while step < total_steps:
        if step < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, float(terminated))

        ep_reward += reward
        ep_torques.append(action.copy())
        ep_displacements.append(max(info.get("x_velocity", 0.0), 0.0))

        obs = next_obs
        step += 1

        if step >= warmup_steps and len(buffer) >= batch_size:
            metrics = agent.update(buffer, batch_size, utd=WAKE_UTD)

            if metrics["diverged"]:
                print(f"[ADRENALINE] step {step}: 정규화 공간 발산 감지 — 비상 그래디언트 클립 작동.")

            if step % log_interval == 0:
                print(
                    f"Step {step:>8d} | "
                    f"critic_loss={metrics['critic_loss']:.4f} | "
                    f"actor_loss={metrics['actor_loss']:.4f} | "
                    f"alpha={metrics['alpha']:.4f} | "
                    f"Qσ={metrics['q_scale']:.2f}"
                )

        if step >= warmup_steps and step % SLEEP_INTERVAL == 0 and len(buffer) >= batch_size:
            print(f"[Sleep] step {step}: 오프라인 공고화 진입 ({SLEEP_UPDATES} bursts).")
            for i in range(SLEEP_UPDATES):
                agent.update(
                    buffer, batch_size,
                    utd=2,
                    weight_decay=SLEEP_WEIGHT_DECAY,
                    update_actor=(i % 2 == 0),
                )
            new_lr = max(agent.alpha_opt.param_groups[0]['lr'] * 0.98, ALPHA_LR_FLOOR)
            agent.alpha_opt.param_groups[0]['lr'] = new_lr
            print(f"[Sleep] 공고화 완료. Qσ={agent.critic.sigma.item():.2f} | alpha_lr={new_lr:.2e}")

        if done:
            cot = compute_cot(ep_torques, ep_displacements)
            if episode % 10 == 0:
                print(f"  Episode {episode:>5d} | step {step:>8d} | "
                      f"ep_reward={ep_reward:>9.2f} | CoT={cot:.4f}")
            ep_reward = 0.0
            ep_torques = []
            ep_displacements = []
            episode += 1
            obs, _ = env.reset()

        if step % eval_interval == 0 and step >= warmup_steps:
            eval_reward, eval_cot = evaluate(agent, eval_env, n_episodes=eval_episodes)
            print(f"\n{'─'*60}\n[EVAL] step={step:>8d} | "
                  f"avg_reward={eval_reward:>9.2f} | avg_CoT={eval_cot:.4f}\n{'─'*60}\n")
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(save_dir)
                print(f"  ✓ New best model saved  (reward={best_eval_reward:.2f})")

    env.close()
    eval_env.close()
    print(f"\n[SNN-APEX] Training complete.  Best eval reward: {best_eval_reward:.2f}")
    return agent


# ─────────────────────────────────────────────
# 8. EVALUATION UTILITY
# ─────────────────────────────────────────────

def evaluate(agent: SACAPEXAgent, env: gym.Env, n_episodes: int = 5) -> tuple:
    total_rewards = []
    total_cots    = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_torques = []
        ep_displacements = []
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            ep_torques.append(action.copy())
            x_vel = info.get("x_velocity", 0.0)
            ep_displacements.append(max(x_vel, 0.0))
        total_rewards.append(ep_reward)
        cot = compute_cot(ep_torques, ep_displacements)
        total_cots.append(cot)

    mean_reward = float(np.mean(total_rewards))
    valid_cots = [c for c in total_cots if not math.isinf(c)]
    mean_cot = float(np.mean(valid_cots)) if valid_cots else float("inf")
    return mean_reward, mean_cot


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(" 🔬 APEX SNN-SAC Training Engine Initialised (α=π fixed)")
    print("=" * 60)

    trained_agent = train(
        env_id="Ant-v4",
        total_steps=200_000,
        batch_size=128,
        hidden_dim=256,
        T=24,
        surrogate_alpha=math.pi,    # ★ 증명된 값
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
