import os
import sys
import numpy as np

ANOMALY_CSV  = "data/anomaly_log_filtered.csv"
INJECTED_CSV = "data/injected_anomalies_filtered.csv"
PIPELINE_ID  = "retail"
WINDOW_SIZE  = 100
STEP_SIZE    = 25
TOTAL_STEPS  = 50_000
MODEL_DIR    = "rl_agent/models/retail"

os.makedirs(f"{MODEL_DIR}/best",        exist_ok=True)
os.makedirs(f"{MODEL_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{MODEL_DIR}/logs",        exist_ok=True)

for f in [ANOMALY_CSV, INJECTED_CSV]:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found")
        sys.exit(1)

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from rl_agent.environment_offline import OfflineAnomalyEnv


class F1LogCallback(BaseCallback):
    def __init__(self, log_every=2000):
        super().__init__()
        self.log_every = log_every
        self._f1s, self._taus = [], []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "f1" in info:
                self._f1s.append(info["f1"])
                self._taus.append(info.get("tau", 2.0))
        if self.n_calls % self.log_every == 0 and self._f1s:
            print(f"  step={self.n_calls:6d} | "
                  f"mean_F1={np.mean(self._f1s):.4f} | "
                  f"mean_τ={np.mean(self._taus):.3f} | "
                  f"min_τ={min(self._taus):.3f} | "
                  f"max_τ={max(self._taus):.3f}")
            self._f1s.clear()
            self._taus.clear()
        return True


def make_env():
    return Monitor(OfflineAnomalyEnv(
        anomaly_csv  = ANOMALY_CSV,
        injected_csv = INJECTED_CSV,
        pipeline_id  = PIPELINE_ID,
        window_size  = WINDOW_SIZE,
        step_size    = STEP_SIZE,
    ))


print("=" * 60)
print("  AdaptiveGuard — DQN Training")
print("=" * 60)

env      = make_env()
eval_env = make_env()

check_env(env.env, warn=True)

# Verify reward variance
obs, _ = env.reset()
rewards = []
for a in range(7):
    obs, r, done, _, info = env.step(a)
    rewards.append(r)
    if done:
        obs, _ = env.reset()
print(f"\nReward variance: {np.var(rewards):.6f}")
if np.var(rewards) < 1e-8:
    print("WARNING: reward is constant — check CSV overlap")
else:
    print("OK: reward varies — training will work")
obs, _ = env.reset()

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path = f"{MODEL_DIR}/best",
    log_path             = f"{MODEL_DIR}/logs",
    eval_freq            = 1000,
    n_eval_episodes      = 5,
    deterministic        = True,
    verbose              = 1,
)
checkpoint_cb = CheckpointCallback(
    save_freq   = 10_000,
    save_path   = f"{MODEL_DIR}/checkpoints",
    name_prefix = "dqn_retail",
    verbose     = 1,
)

model = DQN(
    policy                 = "MlpPolicy",
    env                    = env,
    learning_rate          = 1e-3,
    buffer_size            = 10_000,
    learning_starts        = 200,
    batch_size             = 64,
    tau                    = 0.005,
    gamma                  = 0.95,
    train_freq             = 4,
    target_update_interval = 500,
    exploration_fraction   = 0.40,
    exploration_final_eps  = 0.05,
    policy_kwargs          = dict(net_arch=[64, 64]),
    verbose                = 1,
)

print(f"\nTraining for {TOTAL_STEPS:,} steps...")
model.learn(
    total_timesteps = TOTAL_STEPS,
    callback        = [eval_cb, checkpoint_cb, F1LogCallback()],
    progress_bar    = True,
)

model.save(f"{MODEL_DIR}/dqn_retail_final")
print(f"\nBest model  → {MODEL_DIR}/best/best_model.zip")
print(f"Final model → {MODEL_DIR}/dqn_retail_final.zip")
print("=" * 60)
