# Stability of Recurrent ALIF Spiking Networks under BPTT:
## A Controlled Ablation of Surrogate Slope and Reset-Gradient Detachment

*Working technical note — honestly scoped, single-author undergraduate self-study.*

---

### 1. Summary

I study why a 2-layer recurrent **Adaptive-LIF (ALIF)** spiking network explodes when trained
end-to-end with BPTT over long horizons (T ≈ 200), and I isolate the cause with a clean 2×2
ablation. Two levers control the per-step temporal Jacobian:

1. **Surrogate slope** — the maximum of the arctan surrogate derivative.
2. **Reset-gradient path** — whether the spike-reset term `−z·v_th` carries a gradient
   (`detach_reset=False`) or is stop-gradiented (`detach_reset=True`).

On a Pendulum angular-velocity regression probe, only the configuration with a **unit-bounded
surrogate slope AND a live reset gradient** trains cleanly (R² = 0.88, gradient norm pinned at 1.0).
Every other corner inflates the gradient norm by 11–19 orders of magnitude.

**I want to be explicit about what this is and is not** (Section 5): the slope result re-derives,
from first principles, the design rationale already baked into standard SNN libraries; the
reset-gradient result is a small, preliminary empirical confirmation of a thesis that prior work
(EXODUS) argues on theoretical grounds. The value here is the *methodology and the working system*,
not a claim of new theory.

---

### 2. Setup

- **Task:** supervised regression of Pendulum-v1 angular velocity from `(cosθ, sinθ)` over full
  episodes (T ≈ 200 steps), MSE loss, Adam, grad-clip max-norm 1.0.
- **Model:** `proj(2→64) → ALIF(64) → ALIF(64) → head(64→1)`. ALIF state per step:
  `v ← β·v + (1−β)·I − z·v_th`, `z = Θ(v − v_th)`, adaptive threshold `v_th = 1 + ρ·a`.
- **Surrogate (my parametrization):**
  `f'(x) = a / (π·(1 + (a·x)²))`, so `f'(0) = a/π`.
- **Reference upper bound:** an identical-capacity GRU reaches R² = 0.974 on this probe.

---

### 3. The two stability levers (first-principles)

Unrolled BPTT multiplies a per-step Jacobian T times. If the dominant eigenvalue of that Jacobian
exceeds 1, the product diverges geometrically. Two factors set it:

**(a) Surrogate slope.** The recurrent loop `v → z → rec → v` is multiplied by the surrogate
derivative each step. With my parametrization the peak slope is `a/π`. At the conventional `a = 10`
this is **≈ 3.18 per step** — supercritical; over 200 steps it explodes. Choosing `a = π` sets the
peak slope to **exactly 1.0**, the marginal-stability value.

**(b) Reset gradient.** The reset term `−z·v_th` contributes a *negative* path to ∂v_next/∂v.
Keeping it (`detach_reset=False`) injects negative feedback that opposes the positive recurrent
loop gain; detaching it removes that brake.

---

### 4. Results — 2×2 ablation (single seed, SEED=42)

| Config | surrogate `a` | reset gradient | R² | R²(t>10) | max ‖grad‖ | skipped steps |
|--------|--------------:|:--------------:|------:|---------:|-----------:|--------------:|
| A | 10  | live (False)  | −0.003 | −0.003 | **∞** | 205 (all) |
| B | π   | live (False)  | **0.878** | **0.878** | **1.0** | 0 |
| C | 10  | detached (True) | 0.141 | 0.148 | 1.36×10¹⁹ | 0 |
| D | π   | detached (True) | 0.297 | 0.306 | 4.06×10¹¹ | 0 |

**Reading:**
- **A → B** (drop `a` from 10 to π, reset gradient on): catastrophic ∞ → perfectly bounded 1.0.
  The surrogate slope is the dominant lever.
- **B vs D** (both `a = π`, toggle reset gradient): keeping the reset gradient takes the gradient
  norm from 4×10¹¹ down to **1.0** and R² from 0.30 to **0.88**. At the stable slope, the reset
  gradient acts as the predicted brake.
- The two levers are **not independent**: the reset brake alone (A vs C) cannot rescue a
  supercritical slope — A still diverges to ∞.

---

### 5. Honest positioning vs. prior work

I checked the literature before writing this, and I want to state the limits of novelty plainly:

- **The unit-slope result is not new to the field.** SpikingJelly's default ATan surrogate is
  `α/(2(1+(π/2·α·u)²))`, whose slope at 0 is `α/2`; the library default `α = 2` already yields a
  peak slope of **exactly 1.0**. In other words, the standard default *already* encodes
  "bound the surrogate slope to 1." My `a = π` simply re-derives this for my own parametrization —
  my bug was using `a = 10`, which the first-principles argument correctly diagnoses and fixes.
  This is good debugging, not a new theorem.

- **The reset-gradient result is an empirical echo of EXODUS, against the feedforward convention.**
  There is a real, published tension here. The feedforward convention (snnTorch, SpikingJelly's
  `detach_reset=True`) detaches the reset term because "removing the gradient path through the reset
  terms improves training stability." EXODUS (Bauer et al.) argues the *opposite* for correctness:
  ignoring the reset term causes "numerical instability and incorrect gradients," and notes that
  SLAYER instead "deals with this instability by tweaking a hyperparameter which scales the gradient
  magnitude" — i.e. hand-tuned gradient scaling as a patch for an omitted reset.
  My 2×2 sits exactly on this fault line and supplies a small empirical data point on EXODUS's side,
  in a recurrent ALIF + long-BPTT + RL regime the original work did not test:
  at the stable surrogate scale (`a = π`), **keeping** the reset gradient pins the gradient norm at
  1.0 (R² 0.88), while **detaching** it inflates the norm to 4×10¹¹ (R² 0.30). My surrogate-scale
  knob `a` is precisely the "gradient-magnitude scaling hyperparameter" EXODUS describes — and my
  result shows that, at the boundary, *accounting for the reset beats scaling the surrogate*.
  The **scale × reset interaction** is the part I have not seen stated explicitly: the reset brake
  only rescues training when the surrogate scale is already sub-critical (A, `a=10` + live reset,
  still diverges to ∞). My evidence is preliminary (single seed), but the direction is consistent
  with EXODUS's theory and contrary to the feedforward default.

- **Limitations.** The 2×2 is **single-seed**; the corners differ by orders of magnitude so the
  qualitative ranking is robust, but the exact R² values are not yet multi-seed. The probe is one
  task (Pendulum velocity). I have **not** shown this generalizes across surrogates, neuron models,
  or tasks.

---

### 6. What this demonstrates

A complete, reproducible diagnostic pipeline (step-by-step ablations, finite-gradient guards,
seed control, a GRU upper-bound reference) applied to a real instability in a recurrent spiking
RL system (SNN actor for continuous control). The contribution I would actually stand behind is the
**method and the working system**, plus a correctly-scoped empirical observation that, in recurrent
ALIF networks at the stability boundary, retaining the reset gradient is preferable to the
feedforward-inherited `detach_reset=True` default.

---

*Code and logs: `step4b_probe_fix.py`, `step4b_results.json`, and the surrounding `step*` ablation
suite. Single-author, self-directed study.*
