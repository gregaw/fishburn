# Phase 1 report: the training pipeline

This report records what Phase 1 of the project delivered. It is a snapshot: the code it
describes is tagged `v0.1-phase1` in the project's public repository,
[gregaw/fishburn](https://github.com/gregaw/fishburn), and its results come from one run on
2026-10-10. The [README](../README.md) describes the code as it is now.

## Where Phase 1 sits

The project asks whether training a language model with Nash learning from human feedback
(NLHF), instead of RLHF-style training, changes how well its chain of thought can be
monitored and steered. The [research proposal](https://github.com/gregaw/fishburn/blob/main/Research%20Proposal%20NLHF%20and%20CoT%20monitorability.pdf)
splits the work into three phases:

1. **Phase 1, the training pipeline.** Train the same base model with two methods that
   differ only in how they combine preferences, and check that each one trains.
2. **Phase 2, monitorability.** Measure whether each trained model's chain of thought still
   shows how it reached its answer, as training moves it further from the base model.
3. **Phase 3, controllability.** Measure how well each trained model's reasoning can still
   be steered, by instructions and by steering vectors added to its activations.

Phase 1 is complete. Phases 2 and 3 have not started: the pipeline stages that will hold
their evaluations (`eval_cot` for monitorability, `eval_steering` for controllability) are
placeholders that run, save samples and compute toy metrics (simple counts, such as how
often an answer contains reasoning, that stand in for the real measures).

## Terms used here

- **Base model**: the model every training run starts from, `Qwen/Qwen3-0.6B` with its
  thinking mode off.
- **Leg**: one training method applied to the base model. Phase 1 compares two methods,
  Online DPO (RLHF-style) and NashMP (NLHF), and adds plain DPO as a check that a setup
  trains.
- **Judge**: PairRM (`llm-blender/PairRM`), a small open model that reads a prompt and two
  answers and returns the probability that the first answer is better. Every leg and check
  uses this one judge.
- **Profile**: a config file that sets the model, the data, the legs and their sizes. The
  results below use the `real` profile.
- **Why the methods should differ**: RLHF-style training fits one score per answer, which
  ranks answers like a Borda count and cannot represent a preference cycle (A beats B, B
  beats C, C beats A). NLHF learns from pairwise preferences directly and aims at the
  Maximal Lottery, which can. The two should differ most where preferences conflict.
- **The data**: UltraFeedback, a public dataset of prompts with a preferred (`chosen`) and a
  dispreferred (`rejected`) answer, labelled from GPT-4 scores.

Each trained leg is compared with the base model by one measure of change and three checks:

- **KL from base**: how far training moved the model's answers away from the base model's,
  in nats per answer, estimated every 32 steps on 32 held-out prompts. 0 means unchanged.
  It measures how far a leg moved, not how well it did.
- **Judge check**: both models answer 256 held-out prompts, and the judge gives the
  probability that the trained answer is better, asked in both orders and averaged. The
  score is the mean over prompts.
- **Dataset check**: on 906 held-out UltraFeedback pairs, did training raise the margin
  log p(chosen) − log p(rejected) above the base model's margin? The score is the share of
  pairs where it did.
- **Own-answers check**: did training move the model towards the answers the judge prefers,
  among answers of the kind the model writes? For each of the 256 held-out prompts, the base
  model writes two answers, and the judge (asked in both orders) names the better one the
  winner. The margin is log p(winner) − log p(loser), summed over each answer's tokens. The
  score is the share of answer pairs where the trained model's margin is larger than the
  base model's. That gives 237 pairs: on the other 19 prompts the judge rated the two answers
  exactly even, so there was no winner.

On the three checks, 0.5 is the no-effect value. A model identical to the base scores exactly
0.5 on the dataset and own-answers checks (no margin changes, and a tie counts as half), and
0.5 on average on the judge check. A model can change a lot and still score 0.5, if its
changes do not favour the preferred answers. KL from base has no good or bad value; it is
reported so that legs can be compared at similar amounts of change.

## Planned and delivered

The plan is the project's internal roadmap for Phase 1, which follows the proposal.

| | planned | delivered | why |
| --- | --- | --- | --- |
| NLHF leg | Nash-MD (Nash Mirror Descent), from the TRL library | **NashMP** (Nash Mirror Prox, Tiapkin et al. 2025, [arXiv:2505.19731](https://arxiv.org/abs/2505.19731)), our own PyTorch loop | Nash-MD samples each answer token by token from a mix of the current model and the model before training, which the vLLM inference engine cannot do. NashMP plays against an earlier copy of the model itself, so vLLM can sample for it, which was 3 to 4 times faster end to end than the best transformers `generate` setting in our benchmark ([`vllm.md`](vllm.md)). |
| RLHF-style leg | DPO on the dataset's pairs | **Online DPO**, from TRL; plain **DPO** kept as a check that a setup trains | Online DPO samples its own answers and asks the same judge as NashMP, so the training rule is the only difference between the two. DPO against NashMP would also compare training on fixed data with training on fresh samples. |
| optional legs | PPO with a reward model; SPPO (Self-Play Preference Optimization) | not built | DPO aims at the same target as PPO with a reward model, without training a reward model or running PPO. SPPO moved to later phases. |
| base model | Qwen2.5-0.5B-Instruct | `Qwen/Qwen3-0.6B`, thinking mode off | A newer model of about the same size. Answers are capped at 192 new tokens, which would cut a thought off before the answer, so thinking stays off. |
| labels | the judge relabels every dataset pair, swapping chosen and rejected where it disagrees | the judge's verdict is recorded on every pair, and the `real` profile keeps the dataset's own labels | Only DPO trains on the labels, and the dataset check is scored against them, so the labels are kept as published. The online legs use only the pairs' prompts. |
| checks | loss falls, KL grows, answers are not empty or repetitive, judge check | the same, plus the dataset check and the own-answers check | The dataset check needs no judge, so it can show a leg learning even where the judge sees no difference. |
| KL milestone checkpoints | a checkpoint each time KL from base crosses a set milestone | as planned, at 1, 2, 4, 8 and 16 nats | Later phases compare legs at equal KL, not at equal steps. |
| chain-of-thought evaluators | wired in and run, with no thresholds to pass | as planned | The real evaluators belong to Phases 2 and 3. |

## Results at delivery

The `real` profile ran every stage of all three legs on `Qwen/Qwen3-0.6B`, with one seed, on
one L4 GPU. The run cost $1.51 for all three legs, after a $0.59 canary run (a 16-step run of
the same legs that checks the setup first). DPO made one pass over 2,048 training pairs at 8
per step; the online legs took 96 steps of 8 prompts each.

Each score below is a share with its bootstrap 95 % interval, at the end of training. 0.5 is
the no-effect value, and **bold** marks a score whose whole interval is above 0.5. Train time
is wall-clock time of the training stage, sampling and judging included.

| leg | dataset check | judge check | own-answers check | KL from base (nats) | train time |
| --- | --- | --- | --- | --- | --- |
| DPO, 256 steps | **0.659 [0.628, 0.690]** | 0.520 [0.471, 0.569] | 0.540 [0.477, 0.603] | 0.91 | 27 min |
| Online DPO, 96 steps | 0.509 [0.477, 0.542] | 0.548 [0.499, 0.596] | 0.489 [0.426, 0.557] | 0.89 | 22 min |
| NashMP, 96 steps | 0.499 [0.466, 0.531] | 0.536 [0.491, 0.580] | 0.536 [0.473, 0.603] | 0.95 | 27 min |

- **DPO learns the dataset.** It is the only leg trained on dataset pairs, and it scores 0.659
  on the dataset check. That did not make the judge prefer its answers (0.520).
- **The online legs lean towards the judge's preference, but not beyond noise.** On the judge
  check, Online DPO scores 0.548 and NashMP 0.536, and neither interval is wholly above 0.5.
  This is the judge they trained against, so the check shows whether they learned its
  preference, not whether they got better in general.
- **Neither online leg moves on the dataset pairs.** The judge agrees with the dataset's
  labels on about 83 % of these pairs, so a leg that had learned the judge's preference could
  have moved here too. It did not. Even when the dataset check is scored against the judge's
  pick on each pair instead of the dataset's label, Online DPO scores 0.478 and NashMP 0.472.
- **Their scores overlap on every check**, at about the same KL from base (0.89 and 0.95). At
  this scale (0.6 B parameters, 96 steps, one seed), the comparison shows no difference
  between them yet.
- **Only NashMP reached a KL milestone.** Its KL passed 1 nat during training (it ended at
  0.95), so it saved one milestone checkpoint; the other legs saved none.
- Both online legs take about 14 to 17 s per training step with vLLM, judge included.

## Deferred to later phases

- **The capability floor** (GSM8K and IFEval scores), to Phase 2. A model too weak for a task
  must not be read as having a faithful or unfaithful chain of thought.
- **Contested preferences.** PRISM is a preference dataset whose annotators' backgrounds let
  it be split into groups that disagree, which is where the two methods should differ most.
  Relabelling it with one judge would average that disagreement away, and how to keep it in
  the training signal is still open. It returns with the scale-up below.
- **SPPO** as a required leg.
- **Larger models.** Phase 1 ran at 0.6 B parameters only.

## What Phase 2 needs next

Phase 2 compares the legs along a ladder of checkpoints at rising KL from base, at a larger
scale. The next step, the scale-up, produces that ladder: both online legs at about 1.5 B
parameters, checkpointed at the KL milestones, on UltraFeedback and then on contested data
such as PRISM, with one 7 B confirmation run later. The real monitorability evaluations then
replace the `eval_cot` placeholder.

## Known limits at delivery

- **The chain-of-thought evaluators are placeholders.** `eval_cot` and `eval_steering` save
  samples and toy metrics only.
- **No chain of thought is trained yet.** Thinking mode is off, so the trained models do not
  yet produce the chains of thought the later phases study.
- **No contested preferences yet.** One judge on UltraFeedback; how often PairRM's verdicts
  form cycles on this data was not measured.
- **PairRM is a weak judge for long answers.** It reads only about 400 tokens of prompt and
  answers, and longer text is cut off. It has also seen the training splits of UltraFeedback
  and of HH-RLHF (Anthropic's human-labelled preference dataset).
- **KL milestones need care.** KL is estimated only every 32 steps. In longer runs the first
  estimate can already be past several milestones, which makes those milestone checkpoints the
  same model. Estimate KL more often early in a run to compare legs at equal KL.
- **NashMP has no resume points**, saved training states from which an interrupted run can
  continue, so an interrupted NashMP run starts again. Keep its runs under about an hour.
- **One seed.** Every result above is one seed.
