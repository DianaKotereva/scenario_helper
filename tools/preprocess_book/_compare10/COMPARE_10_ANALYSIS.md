# Compare 10 Chapters: AS-IS vs TO-BE

## Artifacts Produced

### 1) Summaries + per-chapter graph artifacts (10 chapters)
- `tools/preprocess_book/_compare10/asis/summaries` - 10 files
- `tools/preprocess_book/_compare10/asis/results` - 10 `pkl` files
- `tools/preprocess_book/_compare10/asis/graph_nodes` - 10 `pkl` files
- `tools/preprocess_book/_compare10/asis/graph_relations` - 10 `pkl` files

### 2) AS-IS graph
- `tools/preprocess_book/_compare10/asis/runs/bookgraph_asis_10.pkl`

### 3) TO-BE graph
- `tools/preprocess_book/_compare10/tobe/runs/bookgraph_tobe_10.pkl`
- Diagnostics log: `tools/preprocess_book/_compare10/tobe/runs/v2_decisions_tobe_10.jsonl`

---

## High-Level 1:1 Comparison

- AS-IS: `nodes=90`, `relations=77`
- TO-BE: `nodes=86`, `relations=81`

Name diff:
- only in AS-IS (5): `Аулэ`, `Мать Чародея`, `Тень`, `Чертоги Мандоса`, `Чёрные Стражи`
- only in TO-BE (1): `Средиземье (Эндорэ)`

Interpretation:
- TO-BE merges stronger (fewer node names), but also preserves/rebuilds more relation links.

---

## Where We Lose Quality / Merge Behavior

## Stage A - Candidate retrieval (pre-LLM)
- `per_events=169`
- `events_without_candidates=18`
- Of these 18:
  - first-seen names: 17 (normal, often expected)
  - repeated names: 1 (problematic)

Problematic repeated miss:
- `Мандос` (chapter 5): became `термин`, while earlier was `место` -> strict class filter blocked candidates.

Conclusion:
- Candidate stage is not the main merge bottleneck.
- Main loss here is strict class mismatch handling in rare repeats.

## Stage B - LLM judge decision
- `events_with_candidates=151`
- `judge_accept=78`
- `judge_reject_all=73`

For repeated names only:
- follow-up events: 69
- merged (`attach`): 64
- not merged (`create`): 5

Unmerged repeats:
- `Кинжал` (ch3, `judge_reject_all`)
- `Мандос` (ch5, `no_candidates`)
- `Тирион` (ch6, `judge_reject_all`)
- `Аулендур` (ch6/ch9, `judge_reject_all`)

Conclusion:
- For true repeats, merge rate is high (~93%).
- So "low merge" on 10 chapters is not primarily due to judge rejecting all repeats.

## Stage C - Wrong merges (over-merge risk)

Detected `attach` to a different canonical name: 21 cases.

Notable risky examples:
- `Куруфинвэ Феанаро -> Малекит`
- `Тень -> Малекит`
- `Мать Чародея -> Малекит`
- `Чертоги Мандоса -> Мандос`
- `Чёрные Стражи -> Чёрная Гвардия`

Likely cause:
- Current TO-BE accepts first positive judge candidate from top-k without extra disambiguation checks.
- Generic names/titles are too easy to over-attach.

Conclusion:
- Main issue at current stage is precision (false positive merges), not recall.

---

## Classification Drift (source noise)

Observed drift for same names across chapters:
- `Айнур`: `персонаж` / `организация`
- `Аулендур`: `персонаж` / `организация` / `термин`
- `Мандос`: `место` / `термин`
- `Тирион`: `персонаж` / `место`

Impact:
- causes occasional no-candidate or reject outcomes even for repeated names.

---

## Practical Fix Plan (what to treat first)

P0:
1. Do not stop on first accepted candidate in TO-BE.
   - evaluate all accepted candidates in top-k
   - choose best by combined score + stronger consistency checks
2. Add guardrails for generic labels/titles (`Тень`, `Чародей`, `Мать ...`).
   - either block direct merge for generic names
   - or require stronger evidence for merge

P1:
3. Relax strict class filter when name/alias evidence is very strong.
   - especially for repeated names with known class drift
4. Add explicit "same-name follow-up mode":
   - if exact `main_name` seen before, always compare despite class mismatch (with penalty)

P2:
5. Add post-merge audit pass:
   - detect suspicious merges where canonical and merged name are semantically far
   - mark for rollback or re-judge
