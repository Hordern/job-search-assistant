# Claude API cost-optimization notes

Three changes made to this app's scoring pipeline (`scorer.py`), written up
generally so the same levers can be applied to any other Claude-API-backed
app. Each was verified live against this app's real data before being kept.

**Related correctness bug, not just cost — read this even if you don't care
about the bill:** adaptive thinking being on by default (see §1) doesn't
just cost money, it can silently break a call outright. Confirmed three
separate times in this one small codebase (`introspect_scoring_rules.py`,
`term_review.py`, `scorer.py`'s CV-compression call) that with a tight
`max_tokens`, thinking alone can consume the entire budget before the model
writes a single character of its actual answer — `stop_reason: "max_tokens"`,
response content is `['thinking']` only, zero text. If your code does
`"".join(b.text for b in response.content if b.type == "text")` with no
check, you get an **empty string**, not an error — which then either crashes
downstream (a JSON parse on `""`) or, worse, gets silently swallowed by a
fallback (`result = parsed or default`) with no sign anything went wrong.
Rule of thumb: **don't lowball `max_tokens`.** Default to ~16000 for
non-streaming calls unless you have a specific reason to go lower, regardless
of how small you expect the actual answer to be — thinking tokens count
against the same ceiling and you don't control how many it uses.

## 1. Turn off adaptive thinking for deterministic tasks (biggest win)

**The trap:** on current Claude models (Sonnet 5, Opus 5, etc.), simply not
setting the `thinking` parameter does **not** mean "no thinking" — it runs
**adaptive thinking by default**. This is a change from older models, where
omitting `thinking` meant no reasoning at all. The model decides per-request
whether to produce hidden reasoning, and that reasoning is billed as output
tokens — at 5x the input token price (Sonnet 5: $10/MTok output vs $2/MTok
input) — even though `thinking.display` defaults to `"omitted"`, so the
reasoning text is literally invisible in the response unless you ask for it.

If your task is deterministic — structured extraction, classification,
schema-constrained JSON output, anything where the "right answer" doesn't
benefit from visible chain-of-thought — this is pure overhead you're paying
for without knowing it, because nobody is likely logging `response.usage`
to notice.

**Verified here (2026-10-02), real production prompts:**

| Call type | Thinking on (default) | Thinking off | Output token reduction | Output content |
|---|---|---|---|---|
| CV-matching (real job, 22 requirements) | 4,847 tokens | 2,139 tokens | **56% fewer** | identical-length JSON |
| Job-ad extraction (real job) | 4,811 tokens | 2,756 tokens | **43% fewer** | identical-length JSON |

Same schema, same content, same quality — just without paying for invisible
reasoning tokens the task never needed.

**The fix** — explicitly disable it:

```python
response = client.messages.create(
    model=MODEL,
    max_tokens=max_tokens,
    thinking={"type": "disabled"},   # Sonnet 5, Opus 5, etc.
    messages=[...],
)
```

**Model-specific gotcha:** Sonnet 5.5 and Opus 5.5 **reject** `{"type":
"disabled"}` with a 400. On those, use `{"type": "between_tools"}` instead
(valid only at effort `"high"` or below — which is the default anyway).
Pick the right one from the model string before you ship:

```python
def thinking_off_param(model: str) -> dict:
    if model in ("claude-sonnet-5-5", "claude-opus-5-5"):
        return {"type": "between_tools"}
    return {"type": "disabled"}
```

**When NOT to do this:** if the task genuinely benefits from reasoning
(multi-step logic, math, anything where chain-of-thought measurably
improves the answer), don't blanket-disable it — test quality with thinking
on vs off on real examples first. This app's calls are all "extract/classify
against a fixed schema," which is exactly the case where thinking adds cost
with no quality upside. Verify that's actually your case before copying this.

## 2. Prompt caching — put the CV (or any big reused reference doc) first

`cache_control: {"type": "ephemeral"}` on a content block caches everything
**up to and including that block** (~5 min TTL) at ~10% of normal input
price on a later call with a byte-identical prefix (vs ~125% to write it).

**The catch that isn't obvious:** caching only works on a *prefix*. If your
prompt is built as `[variable job-specific stuff] + [static CV/instructions]`
(data first, rules last — often the more natural way to write a prompt),
reordering to `[static CV/instructions] + [variable job-specific stuff]` is
**required**, not optional — a cache breakpoint on static content that comes
*after* variable content never matches, because the "prefix" up to that
point is never identical twice.

This app's CV-matching prompt sends the candidate's CV + fixed instructions
on **every single call** — the single biggest repeated cost in the pipeline,
since the CV is identical for every job scored until the person next updates
it. Restructured so the CV+instructions are a `CachedPrompt.cached` prefix
and the per-job requirements list is `CachedPrompt.variable`, appended after.

**Verified live:** first call of a run writes ~6,100 tokens to cache; every
subsequent call within the TTL reads all ~6,100 tokens from cache instead of
paying full price. In a batch of N jobs scored back-to-back, only the first
pays full CV price — the rest pay ~10%.

**General rule:** look for anything sent unchanged on every call (a system
prompt, a reference document, a fixed instruction/schema block) and always
double-check it's literally the first thing in the request, not buried
after per-call variable content.

## 3. Compress large reference documents that get re-sent every call

If a reference document fed on every call (a CV, a style guide, a product
catalog) is bigger than it needs to be for the model to do its job, every
uncached token in it costs money on every call, and even a cached read still
has a floor cost and a cache-write cost on the first call of each batch/TTL
window.

Built a one-time (not per-call) compression step: when the source document
changes, an LLM call rewrites it to cut redundant phrasing, generic filler,
and over-granular detail from low-relevance sections — while treating every
concrete fact (names, dates, numbers, employers, technologies) as
untouchable and copied verbatim. The compressed version becomes what's
actually sent on every subsequent call; the result is cached via a sha256
hash of the source so recompression only happens when the source actually
changes, never on every run.

**Verified live:** 13,844 → 9,700 chars (30% smaller) on a real update, with
a spot-checked fact ("fluent in French and English") surviving the rewrite
in different wording. This is a one-off cost paid only when the source
document changes — cheap relative to the per-call savings compounding across
every future call.

**General rule:** this only pays off on a document big enough to matter
(roughly >1,500-2,000 tokens — below that, compression risk isn't worth the
marginal saving). A small document should just be used as-is.

---

## Combined effect, measured on this app

All three stack independently and don't conflict:
- Thinking-off cuts **output** token cost (the biggest single lever here,
  since output is priced 5x input and this task never needed reasoning).
- Caching cuts **input** token cost on anything repeated across calls
  within the TTL.
- Compression cuts **the size of what gets cached/sent in the first place**,
  shrinking both the cache-write cost and the cache-miss cost.

None of them required a model change or a quality tradeoff — same model,
same accuracy, verified against real production prompts and real stored
jobs before being kept.
