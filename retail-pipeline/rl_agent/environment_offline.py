import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

ACTION_DELTAS = [-0.30, -0.15, -0.05, 0.0, +0.05, +0.15, +0.30]
N_ACTIONS     = len(ACTION_DELTAS)
STATE_DIM     = 10
SEVERITY_MAP  = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.1}
ANOMALY_TYPES = ["schema_drift", "null_spike", "range_violation", "categorical_drift"]
DEFAULT_TAU   = 2.0
TAU_MIN, TAU_MAX = 0.5, 4.0


class OfflineAnomalyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, anomaly_csv, injected_csv, pipeline_id="retail",
                 window_size=100, step_size=25,
                 tau_min=TAU_MIN, tau_max=TAU_MAX,
                 initial_tau=DEFAULT_TAU, join_seconds=30):
        super().__init__()
        self.pipeline_id  = pipeline_id
        self.window_size  = window_size
        self.step_size    = step_size
        self.tau_min      = tau_min
        self.tau_max      = tau_max
        self.initial_tau  = initial_tau
        self.join_seconds = join_seconds

        df_all = pd.read_csv(anomaly_csv, parse_dates=["detected_at"])
        df_all = df_all[df_all["pipeline_id"] == pipeline_id].copy()
        df_all = df_all.sort_values("detected_at").reset_index(drop=True)

        inj_all = pd.read_csv(injected_csv, parse_dates=["injected_at"])
        inj_all = inj_all[inj_all["pipeline_id"] == pipeline_id].copy()
        inj_all = inj_all.sort_values("injected_at").reset_index(drop=True)

        self.df_anomaly  = df_all
        self.df_injected = inj_all
        self.n_rows      = len(df_all)
        self.n_windows   = max(1, (self.n_rows - window_size) // step_size + 1)

        print(f"[Env] pipeline={pipeline_id} | rows={self.n_rows} | "
              f"injected={len(inj_all)} | windows/ep={self.n_windows}")

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low  = np.full(STATE_DIM, -1.0, dtype=np.float32),
            high = np.full(STATE_DIM,  1.0, dtype=np.float32),
            dtype=np.float32)

        self.cursor   = 0
        self.tau      = initial_tau
        self._prev_f1 = 0.0
        self._step_n  = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cursor   = 0
        self.tau      = self.initial_tau
        self._prev_f1 = 0.0
        self._step_n  = 0
        return self._get_state().astype(np.float32), {}

    def step(self, action):
        self._step_n += 1
        delta    = ACTION_DELTAS[int(action)]
        self.tau = float(np.clip(self.tau + delta, self.tau_min, self.tau_max))
        window   = self._get_window()
        f1, p, r = self._compute_f1(window)
        avg_conf = self._avg_confidence(window)
        reward   = float(np.clip(
            (f1 - self._prev_f1) + 0.10 * avg_conf - 0.05 * abs(self.tau - DEFAULT_TAU),
            -1.0, 1.0))
        self._prev_f1  = f1
        self.cursor   += self.step_size
        obs            = self._get_state()
        terminated     = self.cursor + self.window_size > self.n_rows
        info = {"tau": self.tau, "f1": round(f1, 4),
                "precision": round(p, 4), "recall": round(r, 4),
                "delta": delta, "step": self._step_n}
        return obs.astype(np.float32), reward, terminated, False, info

    def _get_window(self):
        end = min(self.cursor + self.window_size, self.n_rows)
        return self.df_anomaly.iloc[self.cursor:end]

    def _get_state(self):
        window = self._get_window()
        n      = len(window)
        if n == 0:
            return self._neutral_state()
        anomaly_rate = min(n / 100.0, 1.0)
        conf         = window["llm_confidence"].dropna()
        avg_conf     = float(conf.mean()) if len(conf) > 0 else 0.5
        std_conf     = float(conf.std())  if len(conf) > 1 else 0.0
        sev_score    = float(window["severity"].map(SEVERITY_MAP).fillna(0.4).mean())
        type_rates   = [float((window["anomaly_type"] == t).sum()) / n
                        for t in ANOMALY_TYPES]
        mid   = window["detected_at"].iloc[n // 2]
        hour  = mid.hour + mid.minute / 60.0
        angle = 2 * np.pi * hour / 24.0
        state = np.array([anomaly_rate, avg_conf, std_conf, sev_score,
                          *type_rates,
                          float(np.sin(angle)), float(np.cos(angle))],
                         dtype=np.float32)
        return np.clip(state, -1.0, 1.0)

    def _compute_f1(self, window):
        if len(window) == 0:
            return 0.0, 0.0, 0.0
        t_start = window["detected_at"].min() - pd.Timedelta(seconds=self.join_seconds)
        t_end   = window["detected_at"].max() + pd.Timedelta(seconds=self.join_seconds)
        mask    = ((self.df_injected["injected_at"] >= t_start) &
                   (self.df_injected["injected_at"] <= t_end))
        inj_w   = self.df_injected[mask]
        if len(inj_w) == 0:
            return 0.0, 0.0, 0.0
        tp = 0
        matched = set()
        for _, inj in inj_w.iterrows():
            cands = window[
                (window["anomaly_type"] == inj["anomaly_type"]) &
                (abs((window["detected_at"] - inj["injected_at"])
                     .dt.total_seconds()) <= self.join_seconds)]
            if len(cands) > 0:
                best = (abs(cands["detected_at"] - inj["injected_at"])
                        .dt.total_seconds().idxmin())
                if best not in matched:
                    tp += 1
                    matched.add(best)
        fp        = len(window) - len(matched)
        fn        = len(inj_w) - tp
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return float(f1), float(precision), float(recall)

    def _avg_confidence(self, window):
        conf = window["llm_confidence"].dropna()
        return float(conf.mean()) if len(conf) > 0 else 0.5

    def _neutral_state(self):
        return np.array([0.0,0.5,0.0,0.4,0.25,0.25,0.25,0.25,0.0,1.0],
                        dtype=np.float32)
