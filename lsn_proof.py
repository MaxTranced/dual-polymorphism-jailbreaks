"""
Toy model for Constraint Compatibility under a Latent Space Non-Injectivity hypothesis.

Hypothesis (from accompanying post): inputs that look different at the surface can
collapse to the same geometric "shape" at an intermediate layer (non-injectivity).
This toy tests whether an optimizer can (1) match a target hidden state at the
probe layer, (2) keep inputs distinct at the input layer, (3) satisfy a soft
L2 penalty—a fluency proxy (in token-space attacks this would be perplexity).
Distinctness at Layer 0 is not enforced by a loss term; it emerges because many
inputs map to the same Layer-2 activation.

Dual polymorphism (surface vs latent): the optimized input stays distinct at
Layer 0 (surface) but converges to the target representation at Layer 2 (latent).
Single-layer probing is insufficient because the trajectory still differs at
Layer 1; full activation-trajectory monitoring would detect the differing path.

Many-to-one in this toy: default run uses INPUT_DIM=MIDDLE_DIM=16; the bottleneck
is Layer 1→2 (hidden_dim 32 → middle_dim 16). The dimension sweep (RUN_DIMENSION_SWEEP)
additionally explores input_dim > middle_dim (underdetermined input→probe).

Layer convention (used throughout):
  Layer 0 = input space (raw input vector)
  Layer 1 = first hidden activation (after first Linear + ReLU)
  Layer 2 = probe layer (after second Linear + ReLU); the "latent" representation
"""

import torch
import torch.nn as nn

# Hyperparameters (default: input_dim = middle_dim = 16; many-to-one from hidden→probe 32→16)
INPUT_DIM   = 16
HIDDEN_DIM  = 32
MIDDLE_DIM  = 16
OUTPUT_DIM  = 4
LAMBDA      = 0.01   # weight for L2 (fluency-proxy) penalty
LR          = 0.05
N_STEPS     = 500
LOG_EVERY   = 50
N_ROBUSTNESS_SEEDS = 0   # set to e.g. 20 to run multi-seed robustness table
RUN_DIMENSION_SWEEP = False  # set True to run (input_dim, middle_dim) sweep; no CLI
SWEEP_CONFIGS = [(8, 16), (16, 16), (32, 16), (64, 16)]  # (input_dim, middle_dim); fixed HIDDEN_DIM

# Toy network: Layer 0 (input) -> Layer 1 (hidden) -> Layer 2 (probe) -> output
class ToyMLP(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, middle_dim=None, output_dim=None):
        super().__init__()
        in_d = input_dim if input_dim is not None else INPUT_DIM
        h_d = hidden_dim if hidden_dim is not None else HIDDEN_DIM
        m_d = middle_dim if middle_dim is not None else MIDDLE_DIM
        out_d = output_dim if output_dim is not None else OUTPUT_DIM
        self.layer1 = nn.Linear(in_d, h_d)
        self.layer2 = nn.Linear(h_d, m_d)
        self.layer3 = nn.Linear(m_d, out_d)
        self.act = nn.ReLU()

    def forward(self, x):
        h1 = self.act(self.layer1(x))       # Layer 1
        h2 = self.act(self.layer2(h1))       # Layer 2 (probe)
        out = self.layer3(h2)
        return out

    def get_hidden(self, x):
        """Return Layer 2 (probe) activation; no grad (e.g. for target)."""
        with torch.no_grad():
            h1 = self.act(self.layer1(x))
        h2 = self.act(self.layer2(h1))
        return h2

    def get_hidden_grad(self, x):
        """Return Layer 2 activation with grad graph (for optimizing input)."""
        h1 = self.act(self.layer1(x))
        h2 = self.act(self.layer2(h1))
        return h2

    def get_layer1(self, x):
        """Return Layer 1 activation (trajectory diagnostic; supports activation-trajectory monitoring)."""
        with torch.no_grad():
            return self.act(self.layer1(x))

def run_one(seed: int, verbose: bool) -> tuple[float, float, bool, float, float]:
    """One Constraint Compatibility run: match Layer 2, distinct Layer 0, soft L2 penalty. Returns (mse_final, cosine_sim, converged, mse_l0, cos_l0)."""
    torch.manual_seed(seed)
    model = ToyMLP()
    for p in model.parameters():
        p.requires_grad_(False)
    x_target = torch.randn(1, INPUT_DIM)
    with torch.no_grad():
        h_target = model.get_hidden(x_target)   # target "shape" at Layer 2
    x_opt = torch.randn(1, INPUT_DIM, requires_grad=True)
    optimizer = torch.optim.Adam([x_opt], lr=LR)

    if verbose:
        print(f"{'Step':>6} | {'L_activation':>14} | {'L_constraint':>14} | {'L_total':>10}")
        print("-" * 55)

    for step in range(1, N_STEPS + 1):
        optimizer.zero_grad()
        h_opt = model.get_hidden_grad(x_opt)
        L_activation = nn.functional.mse_loss(h_opt, h_target)   # match Layer 2
        L_constraint = x_opt.pow(2).mean()   # soft L2 penalty (fluency proxy; in token space → perplexity)
        L_total = L_activation + LAMBDA * L_constraint
        L_total.backward()
        optimizer.step()
        if verbose and (step % LOG_EVERY == 0 or step == 1):
            print(f"{step:>6} | {L_activation.item():>14.6f} | {L_constraint.item():>14.6f} | {L_total.item():>10.6f}")

    with torch.no_grad():
        h_final = model.get_hidden(x_opt)
    cosine_sim = nn.functional.cosine_similarity(h_final, h_target).item()
    mse_final = nn.functional.mse_loss(h_final, h_target).item()
    converged = cosine_sim > 0.99 and mse_final < 0.01
    # Layer 0 distinctness (surface polymorphism)
    mse_l0 = nn.functional.mse_loss(x_opt, x_target).item()
    cos_l0 = nn.functional.cosine_similarity(x_opt, x_target, dim=1).item()
    return mse_final, cosine_sim, converged, mse_l0, cos_l0


def run_one_config(input_dim: int, middle_dim: int, seed: int = 42) -> tuple[float, float]:
    """One Constraint Compatibility run with given dimensions; return (mse_final, cosine_sim)."""
    torch.manual_seed(seed)
    model = ToyMLP(input_dim=input_dim, middle_dim=middle_dim)
    for p in model.parameters():
        p.requires_grad_(False)
    x_target = torch.randn(1, input_dim)
    with torch.no_grad():
        h_target = model.get_hidden(x_target)
    x_opt = torch.randn(1, input_dim, requires_grad=True)
    optimizer = torch.optim.Adam([x_opt], lr=LR)
    for step in range(1, N_STEPS + 1):
        optimizer.zero_grad()
        h_opt = model.get_hidden_grad(x_opt)
        L_activation = nn.functional.mse_loss(h_opt, h_target)
        L_constraint = x_opt.pow(2).mean()   # fluency proxy (L2)
        (L_activation + LAMBDA * L_constraint).backward()
        optimizer.step()
    with torch.no_grad():
        h_final = model.get_hidden(x_opt)
    cosine_sim = nn.functional.cosine_similarity(h_final, h_target).item()
    mse_final = nn.functional.mse_loss(h_final, h_target).item()
    return mse_final, cosine_sim


if RUN_DIMENSION_SWEEP:
    # Dimension sweep: underdetermined (input > middle) should match easier
    print("Dimension sweep: dim(input) vs dim(probe); fixed middle_dim=16, seed=42")
    print(f"{'input_dim':>10} {'middle_dim':>10} {'MSE':>12} {'cosine':>10} {'converged':>10}")
    print("-" * 52)
    for in_d, mid_d in SWEEP_CONFIGS:
        mse, cos = run_one_config(in_d, mid_d)
        cvd = "yes" if (cos > 0.99 and mse < 0.01) else "no"
        under = " (underdet)" if in_d > mid_d else (" (overdet)" if in_d < mid_d else "")
        print(f"{in_d:>10} {mid_d:>10} {mse:>12.6f} {cos:>10.6f} {cvd:>10}{under}")
    print("-> Matching is easier when input_dim > middle_dim (underdetermined).")

elif N_ROBUSTNESS_SEEDS == 0:
    # Single run: Constraint Compatibility + trajectory (Layer 0/1/2 convention)
    torch.manual_seed(42)
    model = ToyMLP()
    for p in model.parameters():
        p.requires_grad_(False)
    x_target = torch.randn(1, INPUT_DIM)
    with torch.no_grad():
        h_target = model.get_hidden(x_target)
    x_opt = torch.randn(1, INPUT_DIM, requires_grad=True)
    optimizer = torch.optim.Adam([x_opt], lr=LR)

    print(f"{'Step':>6} | {'L_activation':>14} | {'L_constraint':>14} | {'L_total':>10}")
    print("-" * 55)
    for step in range(1, N_STEPS + 1):
        optimizer.zero_grad()
        h_opt = model.get_hidden_grad(x_opt)
        L_activation = nn.functional.mse_loss(h_opt, h_target)
        L_constraint = x_opt.pow(2).mean()   # soft L2 (fluency proxy)
        L_total = L_activation + LAMBDA * L_constraint
        L_total.backward()
        optimizer.step()
        if step % LOG_EVERY == 0 or step == 1:
            print(f"{step:>6} | {L_activation.item():>14.6f} | {L_constraint.item():>14.6f} | {L_total.item():>10.6f}")

    print("\n Final Verification (Constraint Compatibility + dual polymorphism) ")
    with torch.no_grad():
        h_final = model.get_hidden(x_opt)
        h1_target = model.get_layer1(x_target)
        h1_opt = model.get_layer1(x_opt)
    cosine_sim = nn.functional.cosine_similarity(h_final, h_target).item()
    mse_final = nn.functional.mse_loss(h_final, h_target).item()
    print(f"Final MSE between hidden states (Layer 2): {mse_final:.8f}")
    print(f"Cosine similarity (Layer 2): {cosine_sim:.6f}  (1.0 = identical direction)")

    # Layer 0: surface polymorphism — inputs must stay distinct
    mse_l0 = nn.functional.mse_loss(x_opt, x_target).item()
    cos_l0 = nn.functional.cosine_similarity(x_opt, x_target, dim=1).item()
    print("\n--- Layer 0 (input space): inputs stayed distinct (surface) ---")
    print(f"  MSE(input_opt, input_target) = {mse_l0:.6f},  cosine = {cos_l0:.6f}")

    # Layer 1 vs Layer 2: activation-trajectory monitoring (path differs until Layer 2)
    mse_l1 = nn.functional.mse_loss(h1_opt, h1_target).item()
    cos_l1 = nn.functional.cosine_similarity(h1_opt, h1_target, dim=1).item()
    print("\n--- Activation-trajectory: Layer 1 vs Layer 2 ---")
    print(f"  Layer 1:  MSE = {mse_l1:.6f},  cosine = {cos_l1:.6f}  (path still differs)")
    print(f"  Layer 2:  MSE = {mse_final:.6f},  cosine = {cosine_sim:.6f}  (latent polymorphism: converged)")
    print("  -> Single-layer probe sees same shape at Layer 2; full trajectory would differ at Layer 1.")

    CONVERGED = cosine_sim > 0.99 and mse_final < 0.01
    if CONVERGED:
        print("\nConstraint Compatibility: distinct input (Layer 0) converges to target at Layer 2 under L2 penalty.")
        print("(Single run. For robustness across random setups, set N_ROBUSTNESS_SEEDS=20.)")
    else:
        print("\nConvergence incomplete; try more steps or different hyperparameters.")

else:
    # Multi-seed robustness (Constraint Compatibility across random model/target)
    mses, cosines, mses_l0, cosines_l0 = [], [], [], []
    for i in range(N_ROBUSTNESS_SEEDS):
        mse, cos, _, mse_l0, cos_l0 = run_one(seed=i, verbose=False)
        mses.append(mse)
        cosines.append(cos)
        mses_l0.append(mse_l0)
        cosines_l0.append(cos_l0)
    n_converged = sum(1 for m, c in zip(mses, cosines) if c > 0.99 and m < 0.01)
    n_distinct_l0 = sum(1 for c in cosines_l0 if c < 0.99)  # Layer 0 distinct from target (cos < 1)
    mean_mse = sum(mses) / len(mses)
    std_mse = (sum((x - mean_mse) ** 2 for x in mses) / len(mses)) ** 0.5
    mean_cos = sum(cosines) / len(cosines)
    std_cos = (sum((x - mean_cos) ** 2 for x in cosines) / len(cosines)) ** 0.5
    mean_mse_l0 = sum(mses_l0) / len(mses_l0)
    std_mse_l0 = (sum((x - mean_mse_l0) ** 2 for x in mses_l0) / len(mses_l0)) ** 0.5
    mean_cos_l0 = sum(cosines_l0) / len(cosines_l0)
    std_cos_l0 = (sum((x - mean_cos_l0) ** 2 for x in cosines_l0) / len(cosines_l0)) ** 0.5
    print(f"Robustness over {N_ROBUSTNESS_SEEDS} seeds (different model & target each run):")
    print(f"  Layer 2 (probe):  MSE = {mean_mse:.6f} +/- {std_mse:.6f},  cosine = {mean_cos:.6f} +/- {std_cos:.6f}")
    print(f"  Converged (cos>0.99 & MSE<0.01): {n_converged}/{N_ROBUSTNESS_SEEDS}")
    print(f"  Layer 0 (surface): MSE = {mean_mse_l0:.6f} +/- {std_mse_l0:.6f},  cosine = {mean_cos_l0:.6f} +/- {std_cos_l0:.6f}")
    print(f"  Input distinct from target (cos_l0<0.99): {n_distinct_l0}/{N_ROBUSTNESS_SEEDS}")