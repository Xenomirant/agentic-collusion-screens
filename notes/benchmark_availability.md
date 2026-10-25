# Benchmark availability for restricted-disclosure collusion audits

Checked 2026-10-25. Only metadata was fetched: arXiv HTML, GitHub API repo and tree endpoints, HF API dataset and tree endpoints, dataset cards, README/SCHEMA/LICENSE files, and generation code from GitHub raw (to read the output schema). No data files were downloaded. Direct access works for GitHub and arXiv. huggingface.co was reachable only through the proxy (127.0.0.1:8118).

What we need: per-record messages and actions, with actor and recipient IDs (ideally with order or timestamps and object references), plus ground-truth labels that separate a *verified violation* from an *assigned malicious role*, plus benign negatives.

## Summary table

| Dataset | Downloadable? | Host | Size | License | Agents / episode | Episodes | Per-record actor + recipient | Label semantics |
|---|---|---|---|---|---|---|---|---|
| SCHEME (2605.29178) | **No** (no release found) | none | n/a | paper CC BY 4.0; no data | 2-4 | 17 instances × 5 epochs = 85 per model per config, plus topology runs (70 per topology on 14 instances); full total not verified | Yes in the paper's harness (send_message with explicit recipients, timestamps), but not released | Assigned side task (prompt); "sabotage success" = scorer-verified main and side task pass; baseline runs without a side task |
| NARCBench (2604.01151) | **Yes** | HF `aaronrose227/narcbench` + GitHub `aaronrose227/narcbench` | 1.41 GB total; scenario text only about 28 MB | MIT (HF card; GitHub README says MIT but there is no LICENSE file) | 4 (Core/Transfer); 4 in Stego (2 colluders + 2 honest) | Scenario text shipped: gpt-oss-20b Core 68 runs (18 collusion / 50 control), Transfer 31 (13/18), Stego 40 (20/20); qwen3-32b Core 48 (24/24) | Sender in message string `"[Name]: text"`; recipient implicit from channel (public = all 4, private = coalition pair); no timestamps | Assigned role (`colluder`/`honest`) from instructions; `collusion_success` = both colluders voted the target option (outcome check, not a verified harm) |
| Emergent Collusion, Shi et al. (2609.24967), the "Verdict" environment used by NARCBench | **Yes** | HF `SALT-NLP/agent-collusion` + GitHub `SALT-NLP/agent-collusion` | Tables about 81 MB (Parquet); full traces 2,650 gz files = 7.97 GB; peer-cache 75 MB | MIT (LICENSE file on HF and GitHub) | 2 (alice, bob) | 53 conditions, 2,650 trajectories, 27,100 episodes (26,500 evaluation) | **Yes**: `communication` = list of {`event_id`, `round`, `sender`, `receiver`, `content`}; per-agent answers, raw logs, verdicts, private reflections | Emergent (no agent instructed). `collusion` = mutual ACCEPT in constrained conditions, a protocol-verified joint violation; plus LLM-judge `agreement` (explicit coordination in messages, human-validated) and onset pathway EX/RR/SR |
| Colosseum (2602.15198) | **Code only** (official) | GitHub `umass-ai-safety/colosseum` | repo 2.6 MB, code only | no LICENSE file in repo; Terrarium framework MIT | 3-6 (Jira example: 6 agents, 2 colluders) | none released officially | Blackboard posts and tool-call assignments in the harness | Coalition assigned by experimenter; collusion measured by regret against the optimum plus an LLM judge |
| Third-party Colosseum rollouts | Yes (unofficial) | HF `dougalldeepmind/2026-09-03-colosseum-jira-collusion-qwen36-table2-only-9284` (and siblings) | 86 MB, 903 rollout files | none declared | 6 (Jira) | not checked | not checked | LoRA-tuned Qwen3.6 model organisms; provenance unofficial |
| Arbiter (2606.10747) | Partially (results in repo) | GitHub `aisilab/arbiter` | about 20 MB of results JSON | no LICENSE file | multi-agent conversations | not checked | not checked | Assigned misalignment condition (model organism / system prompt / collusion prompt) |

## SCHEME: not available
- I searched the arXiv HTML in full: the only GitHub link is the third-party Inspect Scout tool, and there is no data or code availability statement. I also checked the ICML 2026 workshop page (links only to OpenReview forum s4OoQ4fdOu; OpenReview was blocked by a bot challenge) and ran GitHub repository searches ("SCHEME sabotage", "coordinated sabotage monitoring", "best-laid schemes"). HF dataset search for "scheme" returned nothing relevant.
- What the paper says the logs contain: per-agent step logs of tool calls (code edits) and inter-agent messages. Messages are sent with explicit recipients, constrained by communication permissions, and timestamped (`[HH:MM:SS]`). Topologies are bus and star. Monitors are given everything except reasoning traces.
- Labels: the side task is assigned in prompts. Sabotage success is verified by automated scorers (main and side task both pass). Thresholds come from no-side-task baseline runs.
- Verdict: the ideal record format for us, but unusable unless the authors share the logs. Worth emailing the authors (Radev, Haas, Arnav, Bernabeu-Pérez).

## NARCBench / HELIOS: available, small, instructed
- HF API: `aaronrose227/narcbench`, not gated, MIT, last modified 2026-05-12, 587 files, 1.41 GB. Of this, 1.38 GB is activation `.npz` files for 4 models. Scenario text is under `scenarios/<model>/<tier>/<run>/{run_config.json, results.json, agent_prompts.json}`.
- Discrepancies I noticed:
  - The GitHub README says the data is about 2.6 GB; the HF listing totals 1.41 GB.
  - The HF card's run-count table (for example, Qwen3 Core 31+31, Transfer 36+36, Stego 20+20) does not match the scenario directories actually shipped. Only qwen3_32b/core (48 runs) and gpt_oss_20b core/transfer/stego (68/31/40 runs) have scenario text.
  - The card says scenarios for other models "reused Qwen3's prompts."
  - Read the counts from the files, not the card.
- `results.json` schema (from `generation/core.py`):
  - `private_messages` and `public_messages` are lists of strings formatted `"[AgentName]: text"`;
  - `final_texts[agent]` contains a `RECOMMENDATION: Option X` line;
  - `votes`, `roles`, `collusion_success`, `target_majority`, `colluder_voted_target`, `target_vote_count`.

  Transfer uses the same message fields plus `collusion_active`. `run_config.json` adds `coalition`, `target_option`, `fair_option`, and `scenario_domain`.
- Per run (Core): in collusion runs, one private message per colluder (round 0), then 3 public discussion rounds × 4 agents, then 4 final votes, for about 18 records. There are no timestamps and no object references. The recipient is implied by the channel.
- Labels: role is *assigned by instruction*. `collusion_success` checks that both colluders voted for the target option. That is an outcome consistent with the assigned plan, not an independently verified violation (the target option may even be defensible). `generation/verify_ground_truth.py` audits whether colluders actually used the private channel and followed through. Controls have no coalition.

## Shi et al. "Emergent Collusion" (SALT-NLP): available, best format, 2 agents
- HF API: `SALT-NLP/agent-collusion`, not gated, MIT, last modified 2026-09-24. Code at https://github.com/SALT-NLP/agent-collusion (MIT). The dataset card links a data viewer at https://salt-nlp.github.io/agent-collusion-website/ (not fetched).
- Contents (per README/SCHEMA):
  - `episodes` (27,100 rows): `communication` list of {event_id, round, sender, receiver, content}, plus verdicts, task correctness, rewards, `both_accept`, `collusion`, the joined `agreement` judge label, `trace_file`.
  - `agents` (54,200 rows): task prompt, submitted answer, answer reasoning, task-phase raw log, feedback, private reflection, and the `relaxation` judge.
  - `runs` (2,650 rows): `onset_evaluation_episode`, `onset_pathway` (EX / RR / SR / Other / No onset), `converged_collusion`.
  - `human_*`: 200 adjudicated samples.
  - `traces/`: full model context, tool calls, and reasoning, 7.97 GB.
- Setup: each episode has 2 agents, 5 message rounds, and a 200-character cap per message. The protocol says to ACCEPT only on the peer's complete raw log (median about 5.3k characters), so in constrained conditions the only compliant verdict is REJECT. Mutual ACCEPT is therefore a *verified joint protocol violation* and needs no assigned malicious role. Collusion emerges in 94% of trajectories.
- Useful for Q3:
  - Onset pathways separate explicit coordination (EX) from responsive relaxation (RR) and simultaneous relaxation (SR, no explicit initiator). That is a natural "coordinated vs. independent or shared-error" contrast.
  - The `agreement` judge marks explicit agreements in messages.
  - 21 of 1,216 positive-agreement episodes did *not* end in mutual acceptance, which gives real "abandoned plan" cases.
  - The unconstrained-communication conditions (2 conditions, 100 trajectories) contain legitimate mutual ACCEPTs, which serve as benign negatives. The `collusion` field is null there.
  - Controlled-peer conditions use a scripted Bob.

## Colosseum: no released trajectories
- The official repo (`umass-ai-safety/colosseum`, last pushed 2026-10-23) contains code, configs, and plots only. It has no LICENSE file; the GitHub API reports license = none. Runs write outputs under `experiments/collusion/outputs/<tag>/<timestamp>/`. The paper's results used `terrarium-agents==0.1.1` (tag `paper-v0.1.1`).
- Generating our own runs is possible (Terrarium is MIT; environments are Jira, Hospital, and Meeting/CoLLAB; blackboard posts have known posters and board membership). It needs API keys or a local vLLM model.
- Labels: coalition membership is assigned. Collusion is measured by regret and coalition advantage against a no-secret-channel control, plus a 0-5 LLM judge. "Collusion on paper" (planned but not acted on) exists, so the assigned coalition does not equal a realized violation.
- Unofficial third-party rollouts exist on HF (`dougalldeepmind/...colosseum-jira-collusion...`, 6-agent Jira, LoRA model organisms, no license). Their schema was not checked.

## Recommendation

1. **Primary: `SALT-NLP/agent-collusion`.** It is the only public dataset with explicit per-message `sender`/`receiver`/`event_id`/`round` fields, a *verified* violation label (mutual ACCEPT under an impossible-to-satisfy protocol) rather than an assigned role, emergent behaviour, benign negatives (unconstrained conditions), human-validated coordination labels, and an MIT license. The 81 MB of tables are enough; the 7.97 GB of traces are optional. It maps naturally onto metadata (who messaged whom, when, and which verdict event happened) versus hidden content (message text, verdict, raw log, reflection).
2. **Secondary: NARCBench Core/Transfer (gpt-oss-20b and qwen3-32b scenario text, about 28 MB).** Use it for a 4-agent setting where cross-agent group selection is non-trivial. Labels are instructed roles, and there are only about 190 runs with text.
3. **For more than 2 agents with real action records and object references:** generate episodes with Colosseum/Terrarium (MIT framework) or a SCHEME-like harness, or ask the SCHEME authors for logs.

## Caveats
- **Only 2 agents in SALT.** "Complementary cross-agent groups" collapses to pairs, and a certificate for mutual ACCEPT can be just the two verdict records. To make Q1/Q2 non-trivial there, define the violation as *coordinated* (EX pathway, or agreement judge = yes) rather than just mutual ACCEPT, so the certificate must include communication evidence. Alternatively, treat verdicts as hidden and only "verdict event happened" as metadata.
- **Metadata leakage in NARCBench.** Private-channel messages exist only in collusion runs, so the existence of a private-channel event in public metadata gives the label away. The private channel must be hidden from metadata, or synthetic benign private messages must be added to controls.
- **No object references.** No released dataset has object references (files, tickets, resources) as first-class metadata. SCHEME (code edits) and Colosseum (ticket assignments) would, but SCHEME is unreleased and Colosseum has to be generated.
- **Labels.** NARCBench and Colosseum labels are assigned roles. Use NARCBench's `collusion_success`, or Colosseum's regret, to approximate "verified violation." Neither covers authorized exceptions or misled-agent negatives, so these alternative explanations would have to be constructed (for example, by injecting benign coordination or a misinformed agent into controls).
- **Small samples.** NARCBench scenario text is small (tens of runs per tier). The HF card's counts disagree with the files.
- **License gaps.** The NARCBench GitHub repo, the Colosseum repo, and the Arbiter repo have no LICENSE file. Where a card declares a license (NARCBench HF: MIT), rely on that; otherwise ask before redistributing derived data.
- **Steganography.** NARCBench Stego and the Verdict analysis show that coordination is often not verbalised (2 of 229 collusive episodes). A text-only certificate may not exist for those episodes, so abstention must be allowed and reported.
