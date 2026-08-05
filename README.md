# Fishburn 🐟

Evaluating the Impact of Nash Learning from Human Feedback (NLHF) on Chain-of-Thought Monitorability and Controllability

---

## Proposal & Context

**Fishburn** investigates the interpretability and safety implications of shifting LLM post-training from traditional RLHF to game-theoretic Nash Learning from Human Feedback (NLHF).

* **Read the full proposal:** [Research Proposal: NLHF and CoT Monitorability (PDF)](https://github.com/gregaw/fishburn/blob/main/Research%20Proposal%20NLHF%20and%20CoT%20monitorability.pdf)

---

## Motivation & Core Question

Standard RLHF/DPO acts as a proxy for Borda social choice functions, failing core voting axioms (Condorcet consistency, Majority consistency, IIA) and flattening diverse human preferences.

By framing alignment as a two-player zero-sum game, NLHF approximates **Maximal Lotteries**—preserving preference diversity while meeting these voting axioms. However, game-theoretic optimization may alter internal reasoning mechanics.

> **Core Research Question:**
> *Does replacing scalar Borda rewards (RLHF) with Maximal Lotteries (NLHF) preserve, enhance, or degrade the faithfulness, monitorability, and controllability of Chain-of-Thought (CoT) reasoning?*

---

## Key References

* **[Casper23]** Casper et al., *Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback*, [arXiv:2307.15217](https://www.google.com/search?q=https://arxiv.org/abs/2307.15217)
* **[Halpern26]** Halpern et al., *AI Alignment From Social Choice Perspectives*, [arXiv:2606.21550](https://www.google.com/search?q=https://arxiv.org/abs/2606.21550)
* **[Maura25]** Maura-Rivero et al., *Jackpot! Alignment as a Maximal Lottery*, [arXiv:2501.19266](https://www.google.com/search?q=https://arxiv.org/abs/2501.19266)
* **[Korbak25]** Korbak et al., *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety*, [arXiv:2507.11473](https://www.google.com/search?q=https://arxiv.org/abs/2507.11473)
