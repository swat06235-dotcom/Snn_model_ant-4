"""
================================================================
 STEP 4b-multiseed — 2×2 안정성 어블레이션의 3시드 재현
================================================================
목적: step4b_probe_fix.py 의 단일 시드(SEED=42) 결과를
      3개 시드로 재실행해 "예비 결과"를 "정량 확정"으로 승격.

설계(정직한 통제):
  • 데이터(과제)는 SEED_DATA=42 로 고정 → 모든 시드가 동일 과제를 풂.
  • 모델 초기화 + 미니배치 순서만 SEEDS=[0,1,2] 로 변동
    → 측정하는 것은 "최적화 확률성에 대한 결론의 강건성".
  • 각 조건마다 r2, r2_late, max_grad_norm, skipped, finite 여부 기록.
  • 집계: mean±std (시드 간), 그리고 "성공(R²≥0.5) 시드 수".

판정 기준:
  • B(α=π, reset 살림)가 3시드 모두 R²≥0.8 이고 max‖g‖≈1.0 이면
    → "reset 미분을 살린 안정 설정" 결론이 시드-강건하게 확정.
  • D(α=π, reset 끊음)가 3시드 모두 폭발(‖g‖≫1)이면
    → "detach_reset=True 가 recurrent 에서 해롭다" 도 확정.

실행:  python step4b_multiseed.py   (CPU, 약 5~7분)
================================================================
"""
import os, math, time, json, random, statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

DEVICE = "cpu"
SEED_DATA = 42          # 과제 고정용
SEEDS = [0, 1, 2]       # 모델/학습 확률성 변동용
HID = 64
VEL_SCALE = 8.0
N_TRAIN_EPS, N_TEST_EPS = 150, 30
EPOCHS, MB, LR, MAXNORM = 30, 16, 1e-3, 1.0
GRU_REF_R2 = 0.974
SUCCESS_R2 = 0.8        # "기억 하드웨어 작동" 임계

CONFIGS = [
    ("A_a10_reset",   10.0,    False),
    ("B_aPI_reset",   math.pi, False),
    ("C_a10_detach",  10.0,    True),
    ("D_aPI_detach",  math.pi, True),
]

# ───────── 데이터 (step4b 와 동일) ─────────
def collect_data(n_eps, seed):
    env = gym.make("Pendulum-v1")
    env.action_space.seed(seed)
    X, Y = [], []
    for i in range(n_eps):
        o, _ = env.reset(seed=seed + i)
        xs, ys = [], []
        done = False
        while not done:
            xs.append(o[:2].astype(np.float32))
            ys.append(np.array([o[2] / VEL_SCALE], dtype=np.float32))
            o, r, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
        X.append(xs); Y.append(ys)
    env.close()
    return (torch.as_tensor(np.array(X)), torch.as_tensor(np.array(Y)))

# ───────── SNN (step4b 와 동일) ─────────
class SurrGradSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.save_for_backward(x); ctx.a = a
        return (x > 0).float()
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors; a = ctx.a
        return g * (a / (math.pi * (1.0 + (a * x).pow(2)))), None
def spike(x, a): return SurrGradSpike.apply(x, a)

class ALIFLayer(nn.Module):
    def __init__(self, in_f, out_f, surr, detach_reset, v_th0=1.0):
        super().__init__()
        self.out_f, self.v_th0 = out_f, v_th0
        self.surr, self.detach_reset = surr, detach_reset
        self.ff = nn.Parameter(torch.empty(out_f, in_f)); nn.init.kaiming_uniform_(self.ff, a=math.sqrt(5))
        self.rec = nn.Parameter(torch.empty(out_f, out_f)); nn.init.orthogonal_(self.rec, gain=0.5)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.beta_raw = nn.Parameter(torch.zeros(out_f))
        self.alpha_raw = nn.Parameter(torch.zeros(out_f))
        self.rho = nn.Parameter(torch.full((out_f,), 0.5))
    def tick(self, x, v, a, z):
        beta, alpha = torch.sigmoid(self.beta_raw), torch.sigmoid(self.alpha_raw)
        v_th = self.v_th0 + self.rho * a
        I = F.linear(x, self.ff, self.bias) + F.linear(z, self.rec)
        z_reset = z.detach() if self.detach_reset else z
        vth_reset = v_th.detach() if self.detach_reset else v_th
        v2 = beta * v + (1.0 - beta) * I - z_reset * vth_reset
        z2 = spike(v2 - v_th, self.surr)
        a2 = alpha * a + (1.0 - alpha) * z2
        return z2, v2, a2, z2

class SNNProbe(nn.Module):
    def __init__(self, hid, surr, detach_reset):
        super().__init__()
        self.hid = hid
        self.proj = nn.Linear(2, hid)
        self.l1 = ALIFLayer(hid, hid, surr, detach_reset)
        self.l2 = ALIFLayer(hid, hid, surr, detach_reset)
        self.head = nn.Linear(hid, 1)
    def forward(self, X):
        B, T, D = X.shape
        z = torch.zeros(B, self.hid, device=X.device)
        v1, a1, z1 = z.clone(), z.clone(), z.clone()
        v2, a2, z2 = z.clone(), z.clone(), z.clone()
        outs = []
        for t in range(T):
            cur = self.proj(X[:, t])
            s1, v1, a1, z1 = self.l1.tick(cur, v1, a1, z1)
            s2, v2, a2, z2 = self.l2.tick(s1, v2, a2, z2)
            outs.append(self.head(z2 + 0.1 * torch.tanh(v2)))
        return torch.stack(outs, 1)

# ───────── 학습 (step4b 와 동일) ─────────
def r2_score(pred, target):
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def grads_finite(model):
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True

def train_one(model, Xtr, Ytr, Xte, Yte, dev):
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), LR)
    n = Xtr.shape[0]
    skipped = 0; max_gnorm = 0.0
    for ep in range(1, EPOCHS + 1):
        idx = np.random.permutation(n)
        for s in range(0, n, MB):
            b = idx[s:s+MB]
            x, y = Xtr[b].to(dev), Ytr[b].to(dev)
            loss = F.mse_loss(model(x), y)
            if not torch.isfinite(loss):
                skipped += 1; continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not grads_finite(model):
                skipped += 1; continue
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), MAXNORM))
            max_gnorm = max(max_gnorm, gn)
            opt.step()
    with torch.no_grad():
        pte = model(Xte.to(dev)).cpu()
        r2 = r2_score(pte, Yte)
        r2_late = r2_score(pte[:, 10:], Yte[:, 10:])
    return {"r2": r2, "r2_late": r2_late,
            "max_grad_norm": max_gnorm, "skipped": skipped}

def agg(vals):
    """유한값만 집계. mean/std(시드>=2일 때) 반환."""
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n_finite": 0}
    m = statistics.mean(finite)
    s = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    return {"mean": m, "std": s, "n_finite": len(finite)}

def main():
    dev = torch.device(DEVICE)
    print(f"STEP4b-multiseed | 데이터 고정(SEED={SEED_DATA}), 학습 시드={SEEDS}")
    print("데이터 수집 중...")
    Xtr, Ytr = collect_data(N_TRAIN_EPS, seed=SEED_DATA)
    Xte, Yte = collect_data(N_TEST_EPS, seed=SEED_DATA + 10_000)

    t0 = time.time()
    results = []
    for name, alpha, detach in CONFIGS:
        print(f"\n{'='*60}\n▶ {name}  (α={alpha:.2f}, detach_reset={detach})\n{'='*60}")
        per_seed = []
        for sd in SEEDS:
            torch.manual_seed(sd); np.random.seed(sd); random.seed(sd)
            r = train_one(SNNProbe(HID, alpha, detach), Xtr, Ytr, Xte, Yte, dev)
            r["seed"] = sd
            per_seed.append(r)
            gn = r["max_grad_norm"]
            gn_str = "inf" if not math.isfinite(gn) else f"{gn:.3g}"
            print(f"  seed {sd}: R²={r['r2']:+.4f}  R²(t>10)={r['r2_late']:+.4f}  "
                  f"max‖g‖={gn_str}  skip={r['skipped']}")
        r2s = [r["r2"] for r in per_seed]
        results.append({
            "name": name, "alpha": alpha, "detach_reset": detach,
            "per_seed": per_seed,
            "r2_agg": agg(r2s),
            "r2_late_agg": agg([r["r2_late"] for r in per_seed]),
            "max_grad_norm_per_seed": [r["max_grad_norm"] for r in per_seed],
            "n_success": sum(1 for v in r2s if math.isfinite(v) and v >= SUCCESS_R2),
        })

    dt = time.time() - t0
    print(f"\n{'='*72}\n 3시드 요약  (GRU 상한 R²={GRU_REF_R2}, 성공 임계 R²≥{SUCCESS_R2})\n{'='*72}")
    print(f"{'조건':<16}{'R² mean±std':>18}{'성공/3':>9}{'max‖g‖(시드별)':>30}")
    for r in results:
        a = r["r2_agg"]
        gns = "  ".join("inf" if not math.isfinite(g) else f"{g:.2g}"
                        for g in r["max_grad_norm_per_seed"])
        print(f"{r['name']:<16}{a['mean']:>+10.4f}±{a['std']:<6.4f}"
              f"{r['n_success']:>6d}/3   {gns:>27}")
    print(f"""
{'─'*72}
 판정 가이드
{'─'*72}
 • B 가 3/3 성공 & max‖g‖≈1.0  → "reset 살린 α=π 설정 안정" 시드-강건 확정
 • D 가 3/3 폭발(‖g‖≫1)        → "detach_reset=True 가 recurrent 에서 해롭다" 확정
 • A 가 3/3 비유한(inf)        → "α=10 supercritical 폭발" 확정
 • 결론이 시드별로 뒤집히면    → 정량값 신뢰구간을 이메일에 정직히 명시
 총 소요: {dt/60:.1f}분
""")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "step4b_multiseed_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f" 저장: {out}")

if __name__ == "__main__":
    main()
