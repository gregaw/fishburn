# Fishburn 🐟

Evaluating the Impact of Nash Learning from Human Feedback (NLHF) on Chain-of-Thought Monitorability and Controllability

> **Status: Phase 1 complete.** The training pipeline is built and validated: it trains the same base model with RLHF-style and NLHF-style preference aggregation and runs the same checks on both. See the [code README](code/README.md) for setup and reproduction, and the [Phase 1 report](code/docs/phase1.md) for what was built and tested. Next: the chain-of-thought monitorability and controllability evaluations.

---

## Proposal & Context

**Fishburn** investigates the interpretability and safety implications of shifting LLM post-training from traditional RLHF to game-theoretic Nash Learning from Human Feedback (NLHF).

* **Read the full proposal:** [Research Proposal: NLHF and CoT Monitorability (PDF)](https://github.com/gregaw/fishburn/blob/main/Research%20Proposal%20NLHF%20and%20CoT%20monitorability.pdf)

---

## Code

[`code/`](code/) holds the training pipeline. Its chain-of-thought evaluators are still placeholders. Setup, tests and reproduction steps are in [`code/README.md`](code/README.md).

---

## Motivation & Core Question

Standard RLHF/DPO acts as a proxy for Borda social choice functions, failing core voting axioms (Condorcet consistency, Majority consistency, IIA) and flattening diverse human preferences.

By framing alignment as a two-player zero-sum game, NLHF approximates **Maximal Lotteries**—preserving preference diversity while meeting these voting axioms. However, game-theoretic optimization may alter internal reasoning mechanics.

> **Core Research Question:**
> *Does replacing scalar Borda rewards (RLHF) with Maximal Lotteries (NLHF) preserve, enhance, or degrade the faithfulness, monitorability, and controllability of Chain-of-Thought (CoT) reasoning?*

---

## Key References

* **[Casper23]** Casper et al., *Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback*, [arXiv:2307.15217](https://arxiv.org/abs/2307.15217)
* **[Halpern26]** Halpern et al., *AI Alignment From Social Choice Perspectives*, [arXiv:2606.21550](https://arxiv.org/abs/2606.21550)
* **[Maura25]** Maura-Rivero et al., *Jackpot! Alignment as a Maximal Lottery*, [arXiv:2501.19266](https://arxiv.org/abs/2501.19266)
* **[Korbak25]** Korbak et al., *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety*, [arXiv:2507.11473](https://arxiv.org/abs/2507.11473)
