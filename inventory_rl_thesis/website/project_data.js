const PROJECT_DATA = {
  phase1: {
    policies: [
      { name: "Oracle",  paper_mean: 1474, paper_std: 45,  our_mean: 1455, our_std: 42,  delta: 1.3,  match: true  },
      { name: "A2C",     paper_mean: 870,  paper_std: 67,  our_mean: 829,  our_std: 59,  delta: 4.7,  match: true  },
      { name: "(s,Q)",   paper_mean: 1226, paper_std: 71,  our_mean: 1337, our_std: 49,  delta: 9.1,  match: true  },
      { name: "PPO",     paper_mean: 1213, paper_std: 68,  our_mean: 1602, our_std: 51,  delta: 32.1, match: false },
    ]
  },
  phase2: {
    frequent_short: [
      { name: "(s,Q)",             profit: 758,  service: 0.833, stockout: 0.167, disrpt_cost: 138.8 },
      { name: "PPO Blind",         profit: 1031, service: 0.853, stockout: 0.147, disrpt_cost: 160.0 },
      { name: "PPO Disrpt-Aware",  profit: 398,  service: 0.775, stockout: 0.225, disrpt_cost: 136.1 },
      { name: "PPO LLM-Aug",       profit: 984,  service: 0.857, stockout: 0.143, disrpt_cost: 153.2 },
    ],
    infrequent_long: [
      { name: "(s,Q)",             profit: 460,  service: 0.801, stockout: 0.199, disrpt_cost: 202.1 },
      { name: "PPO Blind",         profit: 726,  service: 0.819, stockout: 0.181, disrpt_cost: 199.5 },
      { name: "PPO Disrpt-Aware",  profit: 689,  service: 0.821, stockout: 0.179, disrpt_cost: 203.4 },
      { name: "PPO LLM-Aug",       profit: 691,  service: 0.821, stockout: 0.179, disrpt_cost: 207.8 },
    ],
    stress_test: [
      { name: "(s,Q)",             profit: -2805, service: 0.452, stockout: 0.548, disrpt_cost: 866.5  },
      { name: "PPO Blind",         profit: -3385, service: 0.533, stockout: 0.467, disrpt_cost: 1737.0 },
      { name: "PPO Disrpt-Aware",  profit: -3385, service: 0.533, stockout: 0.467, disrpt_cost: 1737.0 },
      { name: "PPO LLM-Aug",       profit: -3384, service: 0.533, stockout: 0.467, disrpt_cost: 1737.0 },
    ]
  },
  stats: {
    frequent_short_blind_vs_aware: { t: 5.43, p: 0.0000, significant: true  },
    frequent_short_blind_vs_llm:   { t: 0.61, p: 0.5408, significant: false },
    infrequent_long_all:           { note: "All p > 0.25 — variance dominated" }
  },
  oracle_news: {
    tpr: 95,
    fpr: 2,
    disruption_mean_score: 0.876,
    normal_mean_score: 0.127,
    disruption_headlines: [
      "SEVERE HURRICANE WARNING: Port operations halted indefinitely.",
      "URGENT: Major factory explosion stops all production lines.",
      "CRITICAL ALERT: Key component supplier declares bankruptcy."
    ],
    normal_headlines: [
      "Supply chain operations running smoothly as usual.",
      "Routine maritime shipping continues without issues.",
      "Clear weather and stable markets reported across regions."
    ],
    keywords: {
      "SEVERE": 0.90, "URGENT": 0.88, "CRITICAL": 0.85,
      "shortage": 0.80, "disruption": 0.78, "halted": 0.88,
      "smoothly": -0.30, "stable": -0.25, "routine": -0.20
    }
  }
};
