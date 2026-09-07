import os
import json
import numpy as np
import pandas as pd

ANOMALY_CSV  = "data/anomaly_log_filtered.csv"
INJECTED_CSV = "data/injected_anomalies_filtered.csv"
BEST_MODEL   = "rl_agent/models/retail/best/best_model.zip"
RESULTS_PATH = "rl_agent/models/retail/eval_results.json"
PIPELINE_ID  = "retail"
WINDOW_SIZE  = 100
STEP_SIZE    = 25
N_WINDOWS    = 150
FIXED_TAUS   = [1.0, 1.5, 2.0, 2.5, 3.0]


def make_env():
    from rl_agent.environment_offline import OfflineAnomalyEnv
    return OfflineAnomalyEnv(
        anomaly_csv  = ANOMALY_CSV,
        injected_csv = INJECTED_CSV,
        pipeline_id  = PIPELINE_ID,
        window_size  = WINDOW_SIZE,
        step_size    = STEP_SIZE,
    )


def eval_fixed(tau, n):
    env = make_env()
    obs, _ = env.reset()
    env.tau = tau
    f1s, ps, rs = [], [], []
    for _ in range(n):
        obs, _, done, _, info = env.step(3)
        f1s.append(info["f1"])
        ps.append(info["precision"])
        rs.append(info["recall"])
        if done:
            obs, _ = env.reset()
            env.tau = tau
    return {"tau": tau,
            "mean_f1":   round(float(np.mean(f1s)),  4),
            "std_f1":    round(float(np.std(f1s)),   4),
            "mean_prec": round(float(np.mean(ps)),   4),
            "mean_rec":  round(float(np.mean(rs)),   4),
            "max_f1":    round(float(np.max(f1s)),   4)}


def eval_rl(n):
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from rl_agent.environment_offline import ACTION_DELTAS
    env   = Monitor(make_env())
    model = DQN.load(BEST_MODEL, env=env)
    obs, _ = env.reset()
    f1s, ps, rs, taus, deltas = [], [], [], [], []
    for _ in range(n):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        f1s.append(info["f1"])
        ps.append(info["precision"])
        rs.append(info["recall"])
        taus.append(info["tau"])
        deltas.append(info["delta"])
        if done:
            obs, _ = env.reset()
    action_dist = {}
    for d in ACTION_DELTAS:
        action_dist[f"{d:+.2f}"] = sum(1 for x in deltas if abs(x-d) < 0.001)
    return {"tau":         "adaptive",
            "mean_f1":    round(float(np.mean(f1s)),  4),
            "std_f1":     round(float(np.std(f1s)),   4),
            "mean_prec":  round(float(np.mean(ps)),   4),
            "mean_rec":   round(float(np.mean(rs)),   4),
            "max_f1":     round(float(np.max(f1s)),   4),
            "mean_tau":   round(float(np.mean(taus)), 3),
            "min_tau":    round(float(np.min(taus)),  3),
            "max_tau":    round(float(np.max(taus)),  3),
            "action_dist": action_dist}


def main():
    print("=" * 65)
    print("  AdaptiveGuard — Evaluation Results")
    print("=" * 65)

    for f in [ANOMALY_CSV, INJECTED_CSV]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found")
            return

    df  = pd.read_csv(ANOMALY_CSV)
    inj = pd.read_csv(INJECTED_CSV)
    print(f"Data: {len(df)} anomaly rows · {len(inj)} injection rows\n")

    print(f"{'τ':>6}  {'Mean F1':>8}  {'Std':>6}  "
          f"{'Precision':>10}  {'Recall':>8}  {'Max F1':>7}")
    print("-" * 52)

    fixed_results = []
    best_f1, best_tau = 0.0, 2.0

    for tau in FIXED_TAUS:
        r = eval_fixed(tau, N_WINDOWS)
        fixed_results.append(r)
        mark = "  ← default" if tau == 2.0 else ""
        print(f"{tau:>6.1f}  {r['mean_f1']:>8.4f}  {r['std_f1']:>6.4f}  "
              f"{r['mean_prec']:>10.4f}  {r['mean_rec']:>8.4f}  "
              f"{r['max_f1']:>7.4f}{mark}")
        if r["mean_f1"] > best_f1:
            best_f1  = r["mean_f1"]
            best_tau = tau

    print()
    if os.path.exists(BEST_MODEL):
        rl = eval_rl(N_WINDOWS)
        print(f"{'RL':>6}  {rl['mean_f1']:>8.4f}  {rl['std_f1']:>6.4f}  "
              f"{rl['mean_prec']:>10.4f}  {rl['mean_rec']:>8.4f}  "
              f"{rl['max_f1']:>7.4f}  ← adaptive")
        print(f"         τ range [{rl['min_tau']:.3f}, {rl['max_tau']:.3f}]  "
              f"mean_τ={rl['mean_tau']:.3f}")
        print()
        print("  Action distribution:")
        for delta, count in sorted(rl["action_dist"].items(),
                                   key=lambda x: float(x[0])):
            pct = count / N_WINDOWS * 100
            bar = "█" * int(pct / 3)
            print(f"    Δτ={delta}  {count:>4}x  {pct:>5.1f}%  {bar}")

        default_f1  = next(r["mean_f1"] for r in fixed_results if r["tau"] == 2.0)
        imp_default = (rl["mean_f1"] - default_f1) / (default_f1 + 1e-8) * 100
        imp_best    = (rl["mean_f1"] - best_f1)    / (best_f1    + 1e-8) * 100

        print()
        print("=" * 65)
        print("  THESIS SUMMARY")
        print("=" * 65)
        print(f"  Default baseline (τ=2.0) : F1 = {default_f1:.4f}")
        print(f"  Best fixed (τ={best_tau})   : F1 = {best_f1:.4f}")
        print(f"  RL adaptive agent        : F1 = {rl['mean_f1']:.4f}")
        print(f"  Improvement vs default   : {imp_default:+.2f}%")
        print(f"  Improvement vs best fixed: {imp_best:+.2f}%")
        print(f"  RL Precision = {rl['mean_prec']:.4f}")
        print(f"  RL Recall    = {rl['mean_rec']:.4f}")

        if rl["mean_f1"] > best_f1:
            print("\n  ✓ RL OUTPERFORMS all fixed baselines")
        elif rl["mean_f1"] > default_f1:
            print("\n  ✓ RL outperforms default τ=2.0")
        else:
            print("\n  → Need more training steps")

        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump({"fixed_baselines": fixed_results, "rl_agent": rl,
                       "best_fixed": {"tau": best_tau, "mean_f1": best_f1},
                       "improvements": {"vs_default": round(imp_default, 2),
                                        "vs_best": round(imp_best, 2)}},
                      f, indent=2)
        print(f"\n  Saved → {RESULTS_PATH}")
    else:
        print(f"  No model at {BEST_MODEL}")
        print("  Run train_lightning.py first")

    print("=" * 65)


if __name__ == "__main__":
    main()
