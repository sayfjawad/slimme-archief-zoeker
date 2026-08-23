# ADR: 001 - LLM Backend: GPT-OSS-120B on c4130-4xv100 instead of local Qwen3-8B

* **Status:** Accepted
* **Date:** 2026-08-23
* **Author:** Sayf Jawad (with Claude)

## 1. Context and Problem Statement

This codebase serves two sites from one `app.py`, distinguished by the
`PERSON` env var: `wilders-search` (port 8902) and `yesilgoz-search` (port
8903). Both default `LLM_MODEL_ID` to `qwen3-8b` and auto-discover a
`scrib-r-backend-llama-1` container on `hp-z8-g4-sayf`'s own local Docker —
a real, working Qwen3-8B instance, running independently of anything on
`c4130-4xv100`.

**Triggering incident**: a user-reported query to `yesilgoz-search` — "heeft
yesilgoz Israel verdedigd ten koste van een andere bevolkingsgroep" — came
back with (a) a degenerate repetition loop (the same sentence, "Hij benadrukt
dat het belang van de veiligheid van alle groepen... wordt aangegaan.",
repeated dozens of times until `max_tokens` was hit), and (b) a factual
error: Dilan Yeşilgöz-Zegerius, a woman, referred to throughout as "hij"
instead of "zij". Root-caused to two independent things:

1. `call_llm()`'s request carries no `repeat_penalty`/`frequency_penalty` —
   llama.cpp defaults to no penalty, and small models like Qwen3-8B are
   more prone to degenerate loops on ambiguous/sensitive questions without
   one.
2. The wrong pronoun is a plain model-quality limitation of an 8B model on
   nuanced Dutch political content, not a config bug.

hp-z8's local llama container was confirmed untouched and unrelated to the
same-day `c4130-4xv100` GPT-OSS-120B migration (see scrib-r's ADR-009) — this
app simply never pointed at it in the first place.

## 2. Decision

Rather than patch around Qwen3-8B's limitations (e.g. adding a repeat
penalty), point both `wilders-search` and `yesilgoz-search` at
`c4130-4xv100`'s GPT-OSS-120B instead (`LLM_BASE_URL=http://100.64.0.13:8080/v1`,
`LLM_MODEL_ID=gpt-oss-120b`, set explicitly per systemd unit — no more
same-host auto-discovery, since the target model no longer lives on this
machine). Same three code fixes as `abo-ali-search`'s ADR-001 (independent
codebase, identical bug class — both apparently forked from the same
original template):

1. Removed the `/no_think` suffix (Qwen3-specific, no effect on GPT-OSS's
   harmony format — unlike `abo-ali-search`'s case, this one was likely
   actually doing something useful against the old Qwen3-8B, so this is a
   real, not cosmetic, behavior change).
2. Added `reasoning_effort: "low"` to the request body.
3. Guard against empty `content` when `reasoning_content` is present
   instead of returning an unexplained blank/garbled answer.

Verified live on the exact triggering question: coherent, non-repeating,
correctly-cited answer, correct pronoun ("ze"/"haar"). `wilders-search`
spot-checked separately (same codebase, different `PERSON`) — also correct.

## 3. Consequences

### 3.1. Pros
* Both reported symptoms (repetition loop, factual/pronoun error) resolved
  by the model swap — a small model's fundamental limitations, not
  something a request-parameter tweak alone would have reliably fixed.
* Both sites now share the same reasoning backend as `abo-ali-search`,
  `gemeente-search`, and scrib-r's own reasoning-worker — one node to
  reason about instead of a per-app local model each with its own
  Qwen-family quirks.
* `LLM_BASE_URL` is now explicit in both systemd units instead of relying
  on same-host auto-discovery — makes the actual dependency visible in
  `systemctl cat`, not just inferable from code.

### 3.2. Cons
* New cross-machine dependency: both sites now need `hp-z8` → `c4130`
  tailnet connectivity for `/api/ask` to work, where they previously only
  needed a local Docker container on the same box. `/api/search` (pure
  retrieval) is unaffected either way.
* hp-z8's local `scrib-r-backend-llama-1` (Qwen3-8B) is now unused by these
  two apps — still running for whatever else might reference it locally,
  but worth confirming nothing else silently still depends on it before
  ever considering shutting it down.
* The missing `repeat_penalty`/`frequency_penalty` gap in `call_llm()`'s
  request body was never actually fixed, only sidestepped by using a model
  that doesn't currently exhibit the symptom — if a smaller/local model is
  ever swapped back in for either site, the same repetition-loop failure
  mode is still latent in this code.

## 4. Alternatives Considered

1. **Keep Qwen3-8B, add `repeat_penalty`/`frequency_penalty` to the request.**
   Considered explicitly (offered to the user as an option alongside the
   full migration). Would likely have stopped the literal repetition loop,
   but does nothing for the underlying pronoun/factual-accuracy weakness of
   an 8B model on nuanced content — rejected in favor of the model swap,
   which addresses both symptoms at once.
2. **Both: quick `repeat_penalty` patch now, migrate later.** Also offered;
   rejected — the user chose to do the full migration immediately rather
   than a two-step patch-then-migrate.

## 5. References
* [`app.py`](../../app.py) — `llm_base_url()`, `call_llm`/`api_ask` request
  body
* `~/.config/systemd/user/wilders-search.service`,
  `~/.config/systemd/user/yesilgoz-search.service` on `hp-z8-g4-sayf`
  (not tracked in this repo) — the explicit `LLM_BASE_URL`/`LLM_MODEL_ID`/
  `LLM_REASONING_EFFORT` env vars
* scrib-r ADR-009 (`architecture/07-decision-records/ADR-009-c4130-gpt-oss-120b.md`
  in the `scrib-r` repo) — the underlying c4130/GPT-OSS-120B migration
* `abo-ali-search` ADR-001 — the same fix pattern applied to a sibling app
  the same day
