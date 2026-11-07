# Three-zone incident containment

This experiment asks when short-horizon planning improves an analyst's response
to possible lateral movement while preserving required services. It is a
synthetic, read-only exercise. The CLI emits ranked actions and explanations
for review.

## Scenario and decision order

| Zone | Hosts | Role in the exercise |
|---|---|---|
| Access | Workstation and VPN | Possible initial footholds |
| Operations | Jump host and app | The jump host exposes multiple modeled paths |
| Recovery | Backups | Critical destination |

The base attack graph has directed edges workstation → jump, VPN → jump,
jump → app, jump → backups, and app → backups. An alternate topology adds
workstation → app and VPN → app. The separate service graph requires VPN
reachability to the app via the jump host and app reachability to backups.
In the protected regimes an action that would disconnect either required
flow is unavailable. Relaxed regimes allow disruption and charge for it.
Reachability is a modeled availability proxy.

An episode lasts six abstract ticks. The first observation is a workstation
alert, with subsequent passive observations following a fixed host schedule.
Each tick applies one feasible action, samples at most one attacker spread
event, assesses cost, then returns the next observation. Restrictions and
isolations persist to the end of the episode. Isolating a compromised host
removes its edges while its compromise remains in the state.

| Action | Tokens | Effect |
|---|---:|---|
| Wait | 0 | Preserve the budget and receive the next passive observation |
| Inspect host | 1 | Receive an additional, higher-quality binary observation in noisy mode |
| Restrict directed link | 1 | Remove an eligible link from the attack and service models |
| Isolate host | 2 | Remove that host's incident edges |

In the nominal simulator, a reachable clean destination is added with
probability 0.56 per tick. If several destinations are reachable, their
weights are proportional to the square root of asset value. The faster
attacker uses spread probability 0.72 and weights proportional to squared
value. The planner retains the nominal attacker model in the faster-spread
conditions. The flat control lets the attacker move between all host pairs
while keeping the same list of physically restrictable links. The scenario
allows one possible initial foothold and models spread from it throughout
the episode.

## Belief state and planning

The model enumerates 32 possible compromise sets for five hosts. The prior
puts mass 0.50 on a workstation foothold, 0.20 on a VPN foothold, and 0.30 on
no foothold. Policies in noisy regimes receive a belief updated from the
history of available observations. The passive sensor has sensitivity 0.76
and false positive probability 0.13. Inspections use 0.95 and 0.05. Both
observe the post-spread state. Full-state regimes supply the simulator's
compromise set and remove inspection from the action menu.

For current belief $b$, action $a$, and controls $c$, the planner predicts
$b' = bT_c$ after applying the action's deterministic effect on controls.
Each observation branch reweights that prediction by its likelihood and
normalizes. The objective adds the compromised-value fraction each tick,
0.018 per token spent, and 0.22 per disconnected required flow per tick.
The policy minimizes expected cumulative cost over its horizon, then
replans after the next observation. All policies share the feasible action
rules and the same per-episode budget cap. Reported token use shows how
much of that budget each policy actually spends.

- **Value-first** adapts the original simple allocation baseline to these
  actions. It ranks containment actions using host value and marginal
  compromise estimates, without simulating their consequences.
- **Greedy** minimizes one-tick expected cost.
- **Two-step open-loop** plans two ticks and chooses its hypothetical second
  action without conditioning on the next observation. In actual play it
  still receives new alerts and replans.
- **Two-step adaptive** evaluates the next observation branches and can
  choose a different hypothetical second action for each branch.

In the [ranked example](../results/containment_demo.json), the workstation
alert makes the workstation the most likely lead host. Restricting its link
to the jump host removes three modeled simple paths to critical hosts and
preserves both required flows for one token. Path counts explain structural
effects conditional on the lead host. The reported critical risk is a
separate belief-weighted estimate over the selected horizon assuming no
follow-up action. The ranked cost allows planned follow-up. These quantities
answer different questions.

## Experiment and measured results

The full benchmark varies topology, full versus noisy observation, and
required service protection in a $2 \times 2 \times 2$ design. It then tests
alternate topology and faster spread separately and together, for 11 total
regimes. Each policy receives the same 300 scenario seeds per regime and
the same per-tick random draws for spread and observations. Policies can
produce different realized trajectories because their actions change the
eligible attack paths. Reported intervals use 2,000 percentile bootstrap
resamples of the within-seed cost differences for each regime. These are
scenario-level comparisons under the chosen simulator and cost model.

| Graph, noisy alerts, protected services | Mean episode cost | Critical compromise | Tokens used on average | Required flow-ticks lost |
|---|---:|---:|---:|---:|
| Value-first | 0.950 | 21.0% | 4.00 | 0 |
| One-step greedy | 1.155 | 30.7% | 0.69 | 0 |
| Two-step open-loop | 0.911 | 21.0% | 1.33 | 0 |
| Two-step adaptive | 0.911 | 21.0% | 1.33 | 0 |

The two-step adaptive policy saved **0.244** cost versus greedy in the base
noisy protected scenario (95% paired interval **0.153 to 0.340**).
Critical compromise fell from **30.7% to 21.0%**, with higher token use under
the same four-token cap. The one-token budget sensitivity retained the
critical-compromise difference in this scenario. In the base protected
graph with full state observed, the two-step and greedy policies tied.
Under the combined alternate-topology and faster-spread stress, two-step adaptive
saved **0.337** cost versus greedy (95% interval **0.215 to 0.463**).

Both two-step variants produced the same aggregate results in the reported
runs and neither used inspection. The benchmark supports a lookahead benefit
under noisy observations in these conditions. The value of actively
gathering information remains open. The service mask preserved the required
flows in protected regimes. In the noisy base graph, protection left the
policy results unchanged. In the full-state relaxed graph, the planner
saved 0.159 cost but averaged 1.00 lost flow-ticks versus 0.65 for greedy.
With protection enabled in that full-state graph, both reached zero lost
flow-ticks and tied on mean cost.

- [Full paired summaries](../results/containment_summary.csv)
- [Core budget sensitivity](../results/containment_budget_sweep.csv)
- [Planning advantage figure](../figures/containment_planning.svg)

Critical-compromise rate counts episodes where the app or backups were ever
compromised, with a separate rate conditional on initial foothold. The tail
metric averages the worst 5% of episode-average compromised-value fractions.
Safe containment is recorded only when a foothold was present, no critical
host has been compromised, and no reachable clean host remains. Its timing
is reported alongside the number of safely contained footholds.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m containment recommend
python -m containment benchmark --suite full --episodes 300 --seed 1000 --budget 4 \
  --output results/containment_summary.csv --episodes-out containment_episodes.csv
python -m containment benchmark --suite core --episodes 300 --seed 1000 --budget 1
python -m containment.plot results/containment_summary.csv figures/containment_planning.svg
```

The budget sensitivity file combines `--suite core` runs with budgets 1, 2,
and 4 using the same 300 seeds. Running `recommend --alert negative` shows
how the ranking changes when the first alert is absent.

## Scope of the evidence

The graph, attack probabilities, costs, and alert reliability are synthetic.
The small state space makes the planner auditable and constrains its scale.
The benchmark uses persistent controls, no restoration actions, one modeled
spread opportunity per tick, and a simplified service model. A real pilot
would need authorized topology and incident data, independent tests of the
observation model, and validation of service-impact assumptions. This
release reports a planning comparison and an analyst demo. Training RL on
the containment scenario is a later comparative experiment.
