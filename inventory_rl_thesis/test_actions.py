import numpy as np
from envs.disruption_env import DisruptionEnv
from policies.agents import load_agent
from config import MODELS_DIR

env_blind = DisruptionEnv(disruption_regime='stress_test', include_true_flag=False, include_risk_score=False)
env_aware = DisruptionEnv(disruption_regime='stress_test', include_true_flag=True, include_risk_score=False)
env_llm = DisruptionEnv(disruption_regime='stress_test', include_true_flag=False, include_risk_score=True)

blind = load_agent(MODELS_DIR / "phase2_stress_test_ppo_blind_final", env_blind, "ppo")
aware = load_agent(MODELS_DIR / "phase2_stress_test_ppo_aware_final", env_aware, "ppo")
llm = load_agent(MODELS_DIR / "phase2_stress_test_ppo_llm_final", env_llm, "ppo")

for ep in range(1):
    obs_b, _ = env_blind.reset(seed=42+ep)
    obs_a, _ = env_aware.reset(seed=42+ep)
    obs_l, _ = env_llm.reset(seed=42+ep)

    print(f"--- EP {ep} ---")
    for step in range(5):
        act_b, _ = blind.predict(obs_b, deterministic=True)
        act_a, _ = aware.predict(obs_a, deterministic=True)
        act_l, _ = llm.predict(obs_l, deterministic=True)

        print(f"Step {step}:")
        print(f"  Blind: {act_b}")
        print(f"  Aware: {act_a}")
        print(f"  LLM  : {act_l}")

        obs_b, _, _, _, _ = env_blind.step(act_b)
        obs_a, _, _, _, _ = env_aware.step(act_a)
        obs_l, _, _, _, _ = env_llm.step(act_l)
