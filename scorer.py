"""Anthropic-backed job scoring for Job Search Assistant.

Three-pass design:

  1. build_job_extraction_prompt() — job ad ONLY, no CV, no candidate
     preferences, no candidate eligibility. Extracts atomic requirements
     (with source quotes), eligibility/licence/check requirements as
     structured facts (never candidate-judged), and raw salary/location
     facts. Fully candidate-independent — reusable across profiles.
  2. (conditional) build_audit_prompt() — job ad + the pass-1 requirements
     list, still no CV. Only runs when heuristics suggest pass 1 missed
     something (see needs_audit()) — most jobs skip this.
  3. build_cv_matching_prompt() — the frozen capability requirements list
     (skill/experience/qualification only — not eligibility, not work
     conditions) + the CV. Classifies each requirement's match state.

Everything else — capability_score, salary_fit, location_fit, and every
eligibility result (met/obtainable/action_required/unmet/unknown) — is
computed deterministically in Python from extracted facts and the
candidate's own preferences.json/eligibility config. The model never states
a score, a recommendation, or a judgement about what the candidate holds,
anywhere in the pipeline.

Splitting job-extraction from CV-matching, and eligibility extraction from
eligibility *comparison*, exists specifically to stop candidate context
contaminating what gets extracted, and to stop the model choosing between
several nearby response vocabularies — verified during testing: DeepSeek
returned eligibility-vocabulary and scope-vocabulary values in the
capability match_state field once eligibility items were mixed into that
same array.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Callable

from anthropic import Anthropic
from dotenv import load_dotenv

from profile import BASE_DIR, ENV_PATH, PROFILE, PROFILE_DIR

load_dotenv(ENV_PATH)

CV_PATH = PROFILE_DIR / os.environ.get("CV_PATH", "cv_scoring_full.txt")
# The file you actually paste a fresh/updated CV into. CV_PATH above is the
# derived file actually fed to the model — auto-(re)compressed from this one
# whenever its content changes, see _load_cv(). Not every profile needs
# this: if it's absent, CV_PATH is used directly and unmanaged (legacy mode).
CV_FULL_PATH = PROFILE_DIR / os.environ.get("CV_FULL_PATH", "cv_full.txt")
_CV_HASH_PATH = CV_PATH.with_name(CV_PATH.name + ".hash")
# Below this, compression has nothing meaningful to gain and just adds
# rewrite risk for a CV that was never the cost driver — copy through as-is.
_CV_COMPRESSION_THRESHOLD_CHARS = 8000
PREFS_PATH = PROFILE_DIR / os.environ.get("PREFS_PATH", "preferences.json")
# The scoring methodology (wording, extraction schemas) is shared across
# every profile and lives in one place, separate from personal data — edit
# it without touching Python.
PROMPT_CONFIG_PATH = BASE_DIR / "prompt_config.json"
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")

VALID_IMPORTANCE = {"mandatory", "core", "preferred"}
# "eligibility" is tolerated here (not requested by the schema — see
# job_extraction_instructions rule 8) purely so a provider that still puts a
# licence/check item in the general requirements list (observed: DeepSeek,
# even after a retry) doesn't crash the whole extraction. CAPABILITY_CATEGORIES
# below already excludes it from both capability_score and CV matching either way.
VALID_CATEGORY = {"skill", "experience", "qualification", "work_condition", "eligibility"}
VALID_LEVEL = {"entry", "junior", "mid", "senior", "lead", "manager", "unclear"}
VALID_WORK_ARRANGEMENT = {"onsite", "hybrid", "remote", "unclear"}
VALID_TRAVEL = {"none", "occasional", "regular", "extensive", "unknown"}
VALID_MATCH_STATE = {"direct", "adjacent", "learning", "not_evidenced", "contradicted"}
VALID_EVIDENCE_DEPTH = {"production", "lab", "study", "inferred", "none"}
VALID_SCOPE_MATCH = {"aligned", "moderate_step_down", "severe_step_down", "moderate_step_up", "severe_step_up"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_RECOMMENDATIONS = {"Strong Apply", "Apply", "Stretch", "Pass", "Ineligible", "Excluded"}

VALID_ELIGIBILITY_TIMING = {"must_already_hold", "can_obtain_after_appointment"}
VALID_ELIGIBILITY_RESULT = {"met", "obtainable", "action_required", "unmet", "unknown", "not_applicable"}

IMPORTANCE_WEIGHT = {"mandatory": 5, "core": 3, "preferred": 1}
MATCH_CREDIT = {"direct": 1.0, "adjacent": 0.5, "learning": 0.2, "not_evidenced": 0.0, "contradicted": 0.0}
CAPABILITY_CATEGORIES = {"skill", "experience", "qualification"}

_cv_text: str | None = None
_prefs: dict[str, Any] | None = None
_prompt_config: dict[str, Any] | None = None


@dataclass
class CachedPrompt:
    """A two-part prompt. `cached` is byte-identical across every call that
    builds it within a run (instructions/schema, or — for CV-matching —
    the CV itself plus instructions/schema) and is sent with a
    cache_control breakpoint so Anthropic's prompt cache can reuse it;
    `variable` is the per-call content (the job ad, the per-job
    requirements list) appended after it. The cached half MUST come first
    in the final message — a cache breakpoint only caches a PREFIX, so
    anything static has to be ordered ahead of anything that changes."""

    cached: str
    variable: str

    @property
    def full(self) -> str:
        return self.cached + self.variable

    def __contains__(self, item: str) -> bool:
        return item in self.full


_CV_COMPRESSION_PROMPT = """You are compressing a candidate's master CV for use as an LLM scoring \
input, NOT for human reading — the goal is fewer tokens per job scored, with zero loss of \
anything a scoring model could use as evidence.

Rules, in strict priority order:
1. Never invent, infer, or alter a fact. Every employer, job title, date range, dollar figure, \
percentage, qualification and named tool/technology in the input must appear unchanged in the \
output — copy them verbatim, never paraphrase a number or a name.
2. Never delete a role, a quantified achievement, or a skill/technology keyword. These are the \
only things a scoring model can match against job requirements.
3. You MAY cut: redundant restatements of the same point, generic filler ("results-driven \
professional"), overly granular day-to-day process description that adds no new fact beyond what's \
already stated elsewhere, and verbose phrasing that can be shortened without losing a fact.
4. Older or less senior roles may be compressed harder than recent/primary roles, but every role \
must still appear with its title, employer and dates.
5. If the input is already tight with little redundancy, return it close to unchanged — do not cut \
facts just to hit an arbitrary shorter length.

Return ONLY the compressed CV text. No preamble, no commentary, no markdown code fence, no \
explanation of what you cut.

<cv>
{cv_text}
</cv>"""


def _compress_cv_via_claude(full_text: str) -> str:
    """One-off call (only runs when CV_FULL_PATH's content changes, never
    per-job) that compresses a freshly-pasted full CV for cheaper per-job
    CV-matching calls. Same model as production scoring — fidelity matters
    far more here than cost, since this result becomes the single CV every
    job gets matched against until the next update."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not configured — needed to compress the updated CV")
    client = Anthropic()
    # 16000, not 4096: adaptive thinking (on by default when `thinking` isn't
    # set) counts against max_tokens — confirmed live elsewhere in this
    # codebase that 4096 lets thinking alone consume the whole budget,
    # leaving stop_reason="max_tokens" and zero actual text. Silent here too:
    # the `compressed or full_text` fallback below would mask it by quietly
    # returning the UNCOMPRESSED CV with no error. See CLAUDE_API_COST_NOTES.md.
    response = client.messages.create(
        model=MODEL, max_tokens=16000,
        messages=[{"role": "user", "content": _CV_COMPRESSION_PROMPT.format(cv_text=full_text)}],
    )
    compressed = "".join(b.text for b in response.content if b.type == "text").strip()
    if compressed.startswith("```"):
        compressed = re.sub(r"^```[a-zA-Z]*\n|```$", "", compressed).strip()
    return compressed or full_text


def _load_cv() -> str:
    """Loads the CV actually fed to CV-matching. If CV_FULL_PATH exists,
    treats it as the source of truth: whenever its content changes (tracked
    by a sha256 sidecar next to CV_PATH), automatically regenerates CV_PATH
    — compressed via Claude if it's long enough to be worth it, copied
    through unchanged otherwise — so updating the CV is just overwriting
    CV_FULL_PATH and nothing else. Without CV_FULL_PATH, CV_PATH is read
    directly and unmanaged, for backward compatibility."""
    global _cv_text
    if _cv_text is not None:
        return _cv_text

    if CV_FULL_PATH.exists():
        full_text = CV_FULL_PATH.read_text(encoding="utf-8")
        full_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        cached_hash = _CV_HASH_PATH.read_text(encoding="utf-8").strip() if _CV_HASH_PATH.exists() else None
        if cached_hash == full_hash and CV_PATH.exists():
            _cv_text = CV_PATH.read_text(encoding="utf-8")
            return _cv_text

        if len(full_text) <= _CV_COMPRESSION_THRESHOLD_CHARS:
            compressed = full_text
        else:
            compressed = _compress_cv_via_claude(full_text)
        CV_PATH.write_text(compressed, encoding="utf-8")
        _CV_HASH_PATH.write_text(full_hash, encoding="utf-8")
        _cv_text = compressed
        return _cv_text

    if not CV_PATH.exists():
        raise RuntimeError(f"CV not found at {CV_PATH} — set CV_PATH in .env")
    _cv_text = CV_PATH.read_text(encoding="utf-8")
    return _cv_text


def _load_prefs() -> dict[str, Any]:
    global _prefs
    if _prefs is None:
        if not PREFS_PATH.exists():
            raise RuntimeError(f"Preferences file not found at {PREFS_PATH} — set PREFS_PATH in .env")
        _prefs = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    return _prefs


def load_prefs() -> dict[str, Any]:
    """Public accessor for other modules (e.g. app.py rendering a commute
    badge) that need the candidate's preferences without duplicating the
    load/caching logic."""
    return _load_prefs()


def _load_prompt_config() -> dict[str, Any]:
    global _prompt_config
    if _prompt_config is None:
        if not PROMPT_CONFIG_PATH.exists():
            raise RuntimeError(f"Prompt config not found at {PROMPT_CONFIG_PATH}")
        _prompt_config = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
    return _prompt_config


def _strip_json_fence(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)


def _normalize_for_quote_match(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def _source_haystack(title: str, company: str, description: str) -> str:
    """The exact text the model was shown for this pass (title+company+the
    same [:12000] truncation used in the prompt), normalized for matching."""
    return _normalize_for_quote_match(f"{title} {company} {description[:12000]}")


def _fuzzy_coverage(quote: str, haystack: str) -> float:
    """What fraction of `quote` is covered by contiguous matching runs
    against `haystack` (difflib matching blocks) — unlike bag-of-words
    overlap, this respects word ORDER, so a scrambled paraphrase using the
    same vocabulary won't pass. Both must already be normalized.

    At least one SINGLE matching block must also cover a solid fraction
    (40%) of the quote on its own. A first attempt instead penalized the
    total span between the first and last matched fragment, to stop a
    fabricated quote stitching together two real phrases from widely
    separated parts of the ad — but that also rejected the exact legitimate
    case this fuzzy tier exists for: a shared list lead-in ("You will have
    hands-on knowledge across:") reconstructed onto a specific bullet that
    sits much later in a long bulleted list, genuinely far away from the
    lead-in in the source text. The largest-single-block check instead asks
    "is there at least one substantial real contiguous passage" — the
    specific bullet content alone, ignoring the shared lead-in — which
    passes the legitimate case (that content is one real contiguous quote)
    while still failing a quote made of several comparably-small scattered
    fragments with no single substantial real passage behind it.

    Known residual gap: a deliberately constructed quote stitching two
    large (comparably-sized) real fragments from unrelated parts of the
    document with a fabricated connector between them can still pass, since
    each fragment alone can clear the 40% largest-block bar. Tried
    tightening this (requiring the largest block to dominate total matched
    coverage) and it stopped separating cleanly from the legitimate
    lead-in+bullet case with the real data available. Not fixed further:
    this specific attack shape doesn't match how LLM extraction actually
    fails in practice (hallucination produces paraphrased/invented text
    with low verbatim overlap, not two real spans copy-pasted together with
    fabricated glue) — verified 0 failures across 1104 real source_quotes
    and 705 real CV evidence quotes under this check."""
    if not quote:
        return 0.0
    blocks = [b for b in SequenceMatcher(None, quote, haystack, autojunk=False).get_matching_blocks() if b.size > 0]
    if not blocks:
        return 0.0
    largest = max(b.size for b in blocks)
    if largest < len(quote) * 0.4:
        return 0.0
    return sum(b.size for b in blocks) / len(quote)


def _validate_quote(quote: str, haystack: str, label: str, min_coverage: float = 0.7) -> bool:
    """Three tiers, verified on real data (2026-09-23):
    1. Exact substring (after whitespace/unicode-quote normalization) —
       accept silently. 98% of source_quotes and 57% of CV candidate_evidence
       hit this tier.
    2. Fuzzy contiguous coverage >= min_coverage but not exact — accept, but
       the caller should flag it (this function returns whether it was
       exact). Catches legitimate reformatting (e.g. a shared list lead-in
       reconstructed per split bullet) that a stricter exact-only check
       rejected in testing.
    3. Below min_coverage — reject (raises, caller retries). On real
       candidate_evidence data this caught real fabrications DeepSeek
       labeled "direct" with zero real match: e.g. evidence text that says
       "...but no explicit evidence" or is pure inference/commentary rather
       than a CV excerpt (0.27 coverage) — a pattern that occurred on 4.7%
       of DeepSeek's "direct" matches and 0% of Claude's.
    A plain word-overlap check (ignoring order) was tried first and
    rejected: it would accept a scrambled paraphrase sharing vocabulary
    with the source but not actually stating it."""
    q = _normalize_for_quote_match(quote)
    if not q:
        raise ValueError(f"{label} has an empty or unusable quote")
    if q in haystack:
        return True
    coverage = _fuzzy_coverage(q, haystack)
    if coverage < min_coverage:
        raise ValueError(f"{label} does not appear in the source text ({coverage:.0%} fuzzy coverage): {quote!r}")
    return False


# ---------------------------------------------------------------------------
# Pass 1 — job-only, candidate-independent extraction
# ---------------------------------------------------------------------------

def build_job_extraction_prompt(title: str, company: str, description: str) -> CachedPrompt:
    """Job ad only — no CV, no candidate preferences, no candidate
    eligibility. This extraction is identical regardless of which profile
    will eventually be scored against it.

    Returns a CachedPrompt: system_intro + instructions + schema are
    identical on every call — this prompt runs on every single job, more
    than any other pass — so they're the cached prefix. The untrusted-data
    notice plus the actual job ad text (different every call) is the
    variable part and comes after, which also means the model sees the
    full extraction rules before the untrusted content that follows them,
    not after."""
    cfg = _load_prompt_config()
    schema_json = json.dumps(cfg["job_extraction_schema"], separators=(",", ":"))
    cached = f"""{cfg['system_intro']}

{cfg['job_extraction_instructions']}

Return only a JSON object with exactly this schema. Write it as compact single-line JSON: no indentation, no line breaks between fields, no spaces after colons or commas:
{schema_json}
importance, category, role_level_stated, work_arrangement and travel must use exactly one of the \
listed pipe-separated values each. eligibility_type and timing (inside eligibility_requirements) \
must use exactly one of the listed pipe-separated values each. years_required, salary_min, \
salary_max and onsite_days are integers if stated, otherwise null. salary_text and location_text \
are strings if stated, otherwise null. on_call and contract are booleans."""
    variable = f"""

{cfg['untrusted_data_notice']}
Job title: {title}
Company: {company}
Description/alert text: {description[:12000]}
"""
    return CachedPrompt(cached, variable)


def extract_job_requirements(text: str, title: str, company: str, description: str) -> dict[str, Any]:
    result = json.loads(_strip_json_fence(text))
    if not isinstance(result, dict):
        raise ValueError("Job extraction response is not a JSON object")

    haystack = _source_haystack(title, company, description)
    fuzzy_quotes: list[str] = []
    requirements = result.get("requirements") or []
    if not requirements:
        raise ValueError("Job extraction returned zero requirements — treat as a failed extraction, not a valid result")
    seen_ids: set[str] = set()
    for req in requirements:
        rid = req.get("requirement_id")
        if not rid:
            raise ValueError("Requirement missing requirement_id")
        if rid in seen_ids:
            raise ValueError(f"Duplicate requirement_id: {rid!r}")
        seen_ids.add(rid)
        if req.get("importance") not in VALID_IMPORTANCE:
            raise ValueError(f"Invalid requirement importance: {req.get('importance')!r}")
        if req.get("category") not in VALID_CATEGORY:
            raise ValueError(f"Invalid requirement category: {req.get('category')!r}")
        if not req.get("source_quote"):
            raise ValueError(f"Requirement {rid} has no source_quote")
        if not _validate_quote(req["source_quote"], haystack, f"Requirement {rid} source_quote"):
            fuzzy_quotes.append(rid)

    for item in result.get("eligibility_requirements") or []:
        eid = item.get("eligibility_id")
        if not eid:
            raise ValueError("Eligibility requirement missing eligibility_id")
        if item.get("timing") not in VALID_ELIGIBILITY_TIMING:
            raise ValueError(f"Invalid eligibility timing: {item.get('timing')!r}")
        if not item.get("source_quote"):
            raise ValueError(f"Eligibility requirement {eid} has no source_quote")
        if not _validate_quote(item["source_quote"], haystack, f"Eligibility requirement {eid} source_quote"):
            fuzzy_quotes.append(eid)

    if result.get("role_level_stated") not in VALID_LEVEL:
        raise ValueError(f"Invalid role_level_stated: {result.get('role_level_stated')!r}")
    if result.get("work_arrangement") not in VALID_WORK_ARRANGEMENT:
        raise ValueError(f"Invalid work_arrangement: {result.get('work_arrangement')!r}")
    if result.get("travel") not in VALID_TRAVEL:
        raise ValueError(f"Invalid travel: {result.get('travel')!r}")
    # Not quote-validated (it's a paraphrase, not sourced text) and not
    # required — cosmetic UI copy shouldn't trigger a parse-retry on the
    # already-flaky JSON output if a provider ever omits it.
    if not isinstance(result.get("summary"), str):
        result["summary"] = ""
    # Same reasoning as summary — cosmetic/analytical metadata, not scored,
    # not source-quoted, so a provider omitting it shouldn't trigger a
    # parse-retry.
    if not isinstance(result.get("employer_industry"), str):
        result["employer_industry"] = ""
    result["_fuzzy_quotes"] = fuzzy_quotes
    return result


# ---------------------------------------------------------------------------
# Pass 2 (conditional, or forced for benchmarking) — audit for missed
# requirements
# ---------------------------------------------------------------------------

_STRONG_LANGUAGE_RE = re.compile(r"\b(required|must have|essential|minimum \d+ years?)\b", re.IGNORECASE)


def needs_audit(job_extraction: dict[str, Any], description: str) -> bool:
    """Cheap heuristics for when pass 1 likely missed something — kept
    deliberately conservative so most jobs skip this and cost stays close to
    2 calls/job, not 3, per the "conditional, not compulsory" design."""
    requirements = job_extraction.get("requirements") or []
    core_or_mandatory = [r for r in requirements if r["importance"] in ("mandatory", "core")]

    if not core_or_mandatory:
        return True
    if _STRONG_LANGUAGE_RE.search(description) and not any(r["importance"] == "mandatory" for r in requirements):
        return True
    if len(requirements) < 3 and len(description) > 1000:
        return True
    return False


def build_audit_prompt(title: str, company: str, description: str, job_extraction: dict[str, Any]) -> str:
    cfg = _load_prompt_config()
    existing = json.dumps(
        [{"requirement_id": r["requirement_id"], "requirement": r["requirement"]} for r in job_extraction.get("requirements") or []],
        indent=2,
    )
    next_id_num = len(job_extraction.get("requirements") or []) + 1
    schema_json = json.dumps(cfg["audit_schema"], separators=(",", ":"))
    return f"""{cfg['untrusted_data_notice']}
Job title: {title}
Company: {company}
Full description/alert text: {description[:12000]}

Requirements already extracted:
{existing}

{cfg['audit_instructions']}
Continue requirement_id numbering from R{next_id_num:02d}.

Return only a compact single-line JSON object (no indentation, no line breaks, no spaces after colons or commas) with exactly this schema (empty array if nothing is missing):
{schema_json}
"""


def extract_audit_findings(text: str, title: str, company: str, description: str) -> dict[str, Any]:
    result = json.loads(_strip_json_fence(text))
    if not isinstance(result, dict) or not isinstance(result.get("missing_requirements"), list):
        raise ValueError("Audit response has an invalid schema")
    haystack = _source_haystack(title, company, description)
    fuzzy_quotes: list[str] = []
    for req in result["missing_requirements"]:
        rid = req.get("requirement_id")
        if req.get("importance") not in VALID_IMPORTANCE:
            raise ValueError(f"Invalid audit requirement importance: {req.get('importance')!r}")
        if req.get("category") not in VALID_CATEGORY:
            raise ValueError(f"Invalid audit requirement category: {req.get('category')!r}")
        if not req.get("source_quote"):
            raise ValueError(f"Audit requirement {rid} has no source_quote")
        if not _validate_quote(req["source_quote"], haystack, f"Audit requirement {rid} source_quote"):
            fuzzy_quotes.append(rid)
    result["_fuzzy_quotes"] = fuzzy_quotes
    return result


# ---------------------------------------------------------------------------
# Pass 3 — CV matching against the frozen CAPABILITY requirements only
# (skill/experience/qualification — never eligibility, never work_condition)
# ---------------------------------------------------------------------------

def build_cv_matching_prompt(job_extraction: dict[str, Any]) -> CachedPrompt:
    """Takes the frozen capability requirements list, NOT the original job
    ad text — the CV-matching model never sees the ad, only what pass 1
    decided was a requirement, so it can't independently discover or invent
    gaps. Eligibility and work-condition requirements are deliberately
    excluded here — eligibility is compared in Python against the
    candidate's own config, and work conditions aren't matched against the
    CV at all.

    Returns a CachedPrompt: the CV plus instructions/schema are identical
    on EVERY CV-matching call for this profile until the CV next changes
    (see _load_cv()) — this is the single biggest recurring cost in the
    whole pipeline, since the full CV is otherwise re-sent on every job.
    They're the cached prefix; the per-job role level/family and
    requirements list are the variable part appended after."""
    cfg = _load_prompt_config()
    requirements = [r for r in (job_extraction.get("requirements") or []) if r["category"] in CAPABILITY_CATEGORIES]
    requirements_json = json.dumps(
        [{k: r[k] for k in ("requirement_id", "requirement", "importance", "category")} for r in requirements],
        indent=2,
    )
    schema_json = json.dumps(cfg["cv_matching_schema"], separators=(",", ":"))

    cached = f"""<cv>
{_load_cv()}
</cv>

{cfg['cv_matching_instructions']}

Return only a JSON object with exactly this schema. Write it as compact single-line JSON: no indentation, no line breaks between fields, no spaces after colons or commas:
{schema_json}
match_state, evidence_depth, scope_match and confidence must use exactly one of the listed \
pipe-separated values each. Return one entry in "matches" for every requirement_id listed below \
under "Requirements to evaluate" — no more, no fewer."""
    variable = f"""

{cfg['untrusted_data_notice']}
The role is at level "{job_extraction.get('role_level_stated')}" in the "{job_extraction.get('role_family')}" family.

Requirements to evaluate:
{requirements_json}
"""
    return CachedPrompt(cached, variable)


DEPTH_ALLOWED_BY_STATE = {
    "direct": {"production"},
    "adjacent": {"production", "lab", "study", "inferred"},
    "learning": {"study", "lab"},
    "not_evidenced": {"none"},
    "contradicted": {"production", "lab", "study", "inferred"},
}
# States that must cite a real, quotable CV passage — verified live
# (2026-09-23) that "adjacent" evidence was being left unvalidated: DeepSeek
# could label something adjacent (0.5 credit, same as a real related match)
# with fabricated or absent evidence and it passed every check. Only
# not_evidenced is exempt (candidate_evidence must be null there).
STATES_REQUIRING_EVIDENCE = {"direct", "adjacent", "learning", "contradicted"}


def extract_cv_matches(text: str, expected_ids: set[str]) -> dict[str, Any]:
    result = json.loads(_strip_json_fence(text))
    if not isinstance(result, dict):
        raise ValueError("CV matching response is not a JSON object")
    cv_haystack = _normalize_for_quote_match(_load_cv())
    fuzzy_evidence: list[str] = []
    seen_ids: set[str] = set()
    for m in result.get("matches") or []:
        rid = m.get("requirement_id")
        if rid in seen_ids:
            raise ValueError(f"Duplicate match for requirement_id: {rid!r}")
        seen_ids.add(rid)
        state = m.get("match_state")
        depth = m.get("evidence_depth")
        if state not in VALID_MATCH_STATE:
            raise ValueError(f"Invalid match_state: {state!r}")
        if depth not in VALID_EVIDENCE_DEPTH:
            raise ValueError(f"Invalid evidence_depth: {depth!r}")
        if depth not in DEPTH_ALLOWED_BY_STATE[state]:
            raise ValueError(
                f"Requirement {rid}: evidence_depth {depth!r} is inconsistent with match_state {state!r} "
                f"(allowed: {sorted(DEPTH_ALLOWED_BY_STATE[state])})"
            )
        if state in STATES_REQUIRING_EVIDENCE:
            if not m.get("candidate_evidence"):
                raise ValueError(f"Requirement {rid} claims {state!r} with no candidate_evidence")
            # Verified live (2026-09-23): 4.7% of DeepSeek's "direct" matches quoted
            # analytical commentary or self-admitted inference ("...but no explicit
            # evidence") rather than an actual CV passage — 0% of Claude's did. Applies
            # to adjacent/learning/contradicted too now, not just direct — an
            # unsupported "adjacent" guess was getting the same 0.5 credit as a real one.
            if not _validate_quote(m["candidate_evidence"], cv_haystack, f"Requirement {rid} {state}-match candidate_evidence"):
                fuzzy_evidence.append(rid)
        elif m.get("candidate_evidence"):
            raise ValueError(f"Requirement {rid} is {state!r} but has non-null candidate_evidence — must be null")

    if expected_ids != seen_ids:
        missing = expected_ids - seen_ids
        invented = seen_ids - expected_ids
        raise ValueError(
            f"CV matching must cover exactly the requirement IDs it was given — "
            f"missing: {sorted(missing) or 'none'}, invented/unexpected: {sorted(invented) or 'none'}"
        )

    if result.get("scope_match") not in VALID_SCOPE_MATCH:
        raise ValueError(f"Invalid scope_match: {result.get('scope_match')!r}")
    if result.get("confidence") not in VALID_CONFIDENCE:
        raise ValueError(f"Invalid confidence: {result.get('confidence')!r}")
    if not isinstance(result.get("pitch_angle"), str) or not isinstance(result.get("gaps"), list):
        raise ValueError("pitch_angle/gaps have an invalid type")
    result["_fuzzy_evidence"] = fuzzy_evidence
    return result


# ---------------------------------------------------------------------------
# Deterministic scoring — weighted requirement matrix + guardrail caps
# ---------------------------------------------------------------------------

def compute_capability_score(requirements: list[dict], matches: list[dict]) -> tuple[int, dict[str, Any]]:
    """capability_score is the UNCAPPED weighted average of (importance
    weight × match credit) across capability requirements ONLY
    (skill/experience/qualification) — never a number the model states
    itself, and never touched by eligibility, work conditions, salary or
    location.

    Earlier versions of this function additionally clamped the score itself
    down to a handful of discrete values (5/12/20/30) whenever a couple of
    core/mandatory requirements were unmatched. Verified against a 20-job
    Claude-vs-DeepSeek run (2026-09-23): since 2-3+ unmatched core items is
    close to universal on a real job ad, that cap fired on nearly every job
    and collapsed a 15-89 point range of genuine evidence differences down
    to 3 buckets — it wasn't a guardrail, it was overwriting the signal
    (production Claude scores on the same jobs averaged 30.6 points higher
    before this capping existed). Disqualifiers are real, but they belong
    in the recommendation TIER (see compute_final_result), not stamped over
    the score. Returns (uncapped_score, decision_flags)."""
    capability_requirements = [r for r in requirements if r["category"] in CAPABILITY_CATEGORIES]
    if not capability_requirements:
        return 50, {"unreliable": "no capability requirements were extracted"}

    match_by_id = {m["requirement_id"]: m for m in matches}
    total_weight = 0.0
    earned = 0.0
    for req in capability_requirements:
        weight = IMPORTANCE_WEIGHT.get(req["importance"], 1)
        total_weight += weight
        match = match_by_id.get(req["requirement_id"])
        credit = MATCH_CREDIT.get(match["match_state"], 0.0) if match else 0.0
        earned += weight * credit
    score = round(100 * earned / total_weight) if total_weight else 50

    def unmatched(importance: str) -> list[dict]:
        return [
            r for r in capability_requirements
            if r["importance"] == importance
            and match_by_id.get(r["requirement_id"], {}).get("match_state") in ("not_evidenced", "contradicted")
        ]

    unmatched_mandatory = unmatched("mandatory")
    unmatched_core = unmatched("core")
    decision_flags = {
        "unmatched_mandatory": [r["requirement"] for r in unmatched_mandatory],
        "unmatched_core_count": len(unmatched_core),
    }

    return max(0, min(100, score)), decision_flags


# ---------------------------------------------------------------------------
# Deterministic salary/location fit — computed from raw extracted facts,
# never asked of the model, so Pass 1 stays candidate-independent.
# ---------------------------------------------------------------------------

def compute_salary_fit(salary_min: int | None, salary_max: int | None, prefs: dict[str, Any]) -> str:
    if salary_min is None and salary_max is None:
        return "unknown"
    sal = prefs.get("salary", {})
    if salary_min is not None and salary_max is not None:
        value = (salary_min + salary_max) / 2
    else:
        value = salary_min if salary_min is not None else salary_max
    if value >= sal.get("great_floor_aud", float("inf")):
        return "great"
    if value >= sal.get("good_floor_aud", float("inf")):
        return "good"
    if value >= sal.get("consider_floor_aud", float("inf")):
        return "consider"
    return "poor"


def compute_location_fit(location_text: str | None, work_arrangement: str, onsite_days: int | None, prefs: dict[str, Any]) -> str:
    loc = prefs.get("location", {})
    great_areas = [a.lower() for a in loc.get("great_areas", [])]
    consider_areas = [a.lower() for a in loc.get("consider_areas", [])]
    min_wfh = loc.get("min_wfh_days_for_hybrid_ok")
    text = (location_text or "").lower()
    in_great = any(a in text for a in great_areas)
    in_consider = any(a in text for a in consider_areas)

    if work_arrangement == "remote":
        return "great"
    if in_great:
        return "great"
    if work_arrangement == "hybrid" and min_wfh is not None and onsite_days is not None and (5 - onsite_days) >= min_wfh:
        return "great"
    if in_consider:
        return "ok"
    if work_arrangement == "hybrid":
        return "ok"
    if work_arrangement == "unclear" or (location_text is None and work_arrangement == "onsite" and onsite_days is None):
        return "unknown" if location_text is None else "poor"
    if work_arrangement == "onsite":
        return "poor"
    return "unknown"


# ---------------------------------------------------------------------------
# Deterministic eligibility — Python compares the extracted requirement
# against the candidate's own declared profile. The model never judges
# whether the candidate holds anything.
# ---------------------------------------------------------------------------

def _aggregate_eligibility_requirements(eligibility_requirements: list[dict]) -> dict[str, dict]:
    """If the same eligibility_type is extracted more than once for a job
    (e.g. mentioned in both a summary line and a detailed requirements
    section), merge to the STRICTEST combination — mandatory over optional,
    must_already_hold over can_obtain_after_appointment — rather than
    letting whichever occurrence came last silently win. Verified live
    (2026-09-23): compute_eligibility() used to key results by type in a
    plain dict, so a second, weaker extraction of the same type (e.g.
    correctly mandatory the first time, mistakenly "preferred" the second)
    would overwrite the first and turn a real "unmet" into "action_required"."""
    merged: dict[str, dict] = {}
    for item in eligibility_requirements:
        etype = item.get("eligibility_type", "other")
        if etype not in merged:
            merged[etype] = dict(item)
            continue
        existing = merged[etype]
        existing["mandatory"] = bool(existing.get("mandatory")) or bool(item.get("mandatory"))
        if item.get("timing") == "must_already_hold":
            existing["timing"] = "must_already_hold"
    return merged


def compute_eligibility(eligibility_requirements: list[dict], profile_eligibility: dict[str, Any]) -> dict[str, str]:
    """Returns {eligibility_type: status}. status is one of
    met | obtainable | action_required | unmet | unknown, per item actually
    extracted from the ad (not_applicable is implicit for anything not
    extracted at all — the job simply didn't ask for it)."""
    results: dict[str, str] = {}
    for etype, item in _aggregate_eligibility_requirements(eligibility_requirements).items():
        person = (profile_eligibility or {}).get(etype)
        if person is None or person.get("has") is None:
            results[etype] = "unknown"
            continue
        has = person.get("has")
        willing = bool(person.get("willing_to_obtain"))
        mandatory = item.get("mandatory", True)
        must_already_hold = item.get("timing") == "must_already_hold"

        if has:
            results[etype] = "met"
        elif must_already_hold:
            if willing:
                results[etype] = "action_required"
            else:
                results[etype] = "unmet" if mandatory else "action_required"
        else:  # can_obtain_after_appointment, has is False
            if willing:
                results[etype] = "obtainable"
            else:
                results[etype] = "unmet" if mandatory else "action_required"
    return results


def compute_industry_fit(employer_industry: str | None, prefs: dict[str, Any]) -> tuple[str | None, str | None]:
    """Returns (status, matched_label) — status is 'excluded', 'downmarked',
    or None. employer_industry is Claude's free-text short label (e.g.
    "Banking", "Retail Banking"), not a fixed enum, so matching is a
    case-insensitive substring check in either direction against the
    configured list rather than exact equality, which would miss obvious
    matches like "Banking" vs "Retail Banking"."""
    if not employer_industry:
        return None, None
    industry_prefs = prefs.get("industry_preferences") or {}
    text = employer_industry.lower()
    for label in industry_prefs.get("exclude", []):
        if label.lower() in text or text in label.lower():
            return "excluded", label
    for label in industry_prefs.get("downmark", []):
        if label.lower() in text or text in label.lower():
            return "downmarked", label
    return None, None


def compute_final_result(job_extraction: dict[str, Any], cv_matching: dict[str, Any]) -> dict[str, Any]:
    """Combine pass-1 and pass-3 output into the final result, computing
    every fit/eligibility dimension deterministically in Python.

    capability_score is the uncapped weighted-evidence score (see
    compute_capability_score) — it is never touched by eligibility, work
    conditions, salary, location, or scope. Disqualifiers instead demote the
    recommendation TIER, so "good evidence but missing one mandatory thing"
    and "no evidence at all" stay visibly different scores even though both
    might land on the same recommendation.

    scope_match is asymmetric on purpose: a severe step-UP (role clearly
    above the candidate's demonstrated level) is a real capability risk and
    demotes the tier; a severe step-DOWN (role below their level) is a
    retention/fit question, not evidence they can't do it — it's surfaced
    as an informational flag only, never a tier penalty.

    An unmet eligibility item (something required that the candidate
    neither has nor will get) produces a distinct "Ineligible" outcome, not
    just the lowest normal tier — the job is genuinely inapplicable
    regardless of skill fit, which is a different situation from "a poor
    skills match". capability_score itself is untouched even then, so it
    still ranks "how good a fit skill-wise" if the blocker is ever
    resolved."""
    requirements = job_extraction.get("requirements") or []
    matches = cv_matching.get("matches") or []
    scope_match = cv_matching.get("scope_match", "aligned")
    prefs = _load_prefs()

    salary_fit = compute_salary_fit(job_extraction.get("salary_min"), job_extraction.get("salary_max"), prefs)
    location_fit = compute_location_fit(
        job_extraction.get("location_text"), job_extraction.get("work_arrangement"), job_extraction.get("onsite_days"), prefs
    )
    industry_status, industry_label = compute_industry_fit(job_extraction.get("employer_industry"), prefs)

    capability_score, decision_flags = compute_capability_score(requirements, matches)
    decision_flags["scope_match"] = scope_match

    eligibility = compute_eligibility(job_extraction.get("eligibility_requirements") or [], prefs.get("eligibility", {}))
    unmet_eligibility = [k for k, v in eligibility.items() if v == "unmet"]
    action_required_eligibility = [k for k, v in eligibility.items() if v == "action_required"]

    # Base tier from the uncapped capability_score alone.
    if capability_score >= 75:
        recommendation = "Strong Apply"
    elif capability_score >= 55:
        recommendation = "Apply"
    elif capability_score >= 30:
        recommendation = "Stretch"
    else:
        recommendation = "Pass"

    tier_rank = {"Pass": 0, "Stretch": 1, "Apply": 2, "Strong Apply": 3}
    demotion_reasons: list[str] = []
    info_flags: list[str] = []

    def demote(tier: str, reason: str) -> None:
        nonlocal recommendation
        if tier_rank[tier] < tier_rank[recommendation]:
            recommendation = tier
        demotion_reasons.append(reason)

    if decision_flags.get("unmatched_mandatory"):
        demote("Stretch", f"unmatched mandatory requirement: {decision_flags['unmatched_mandatory'][0]}")
    if decision_flags.get("unmatched_core_count", 0) >= 3:
        demote("Stretch", f"{decision_flags['unmatched_core_count']} unmatched core requirements")
    if scope_match == "severe_step_up":
        demote("Stretch", "severe scope mismatch: severe_step_up")
    elif scope_match == "moderate_step_up":
        demote("Apply", "moderate_step_up")
    elif scope_match in ("severe_step_down", "moderate_step_down"):
        info_flags.append(f"{scope_match} — role likely below your demonstrated level; a fit/retention question, not a capability gap")
    if salary_fit == "poor" or location_fit == "poor":
        demote("Stretch", "poor salary or location fit")
    if industry_status == "downmarked":
        demote("Stretch", f"down-ranked industry: {industry_label}")
    if action_required_eligibility:
        demote("Stretch", f"eligibility action required ({', '.join(action_required_eligibility)})")
    # Unknown (the ad simply didn't state it — the common case) is visible
    # uncertainty, not evidence of a poor fit — it must never demote the
    # tier. Earlier versions capped this at "Apply", which punished the
    # majority of ads that just don't list a salary. Surfaced as an info
    # flag instead so it's still visible on the card.
    if salary_fit == "unknown" or location_fit == "unknown":
        info_flags.append("salary or location not stated in the ad — unverified, not a poor fit")

    if unmet_eligibility:
        recommendation = "Ineligible"
        demotion_reasons.append(f"unmet eligibility requirement ({unmet_eligibility[0]}) — job is not applicable regardless of skill fit")

    # Industry exclusion overrides everything except eligibility — a strong
    # personal preference ("I don't want to work in banking"), not a
    # capability judgement, so it takes priority over tier demotions from
    # scope/salary/location but doesn't claim you're literally unable to
    # take the job the way unmet eligibility does.
    if industry_status == "excluded" and recommendation != "Ineligible":
        recommendation = "Excluded"
        demotion_reasons.append(f"excluded industry: {industry_label} — job matches an employer type you've asked to exclude, regardless of role fit")

    return {
        "cv_fit_score": capability_score,
        "capability_score": capability_score,
        "salary_fit": salary_fit,
        "location_fit": location_fit,
        "industry_fit": {"status": industry_status, "label": industry_label},
        "eligibility": eligibility,
        "overall_score": capability_score,
        "recommendation": recommendation,
        "pitch_angle": cv_matching.get("pitch_angle", ""),
        "gaps": cv_matching.get("gaps", []),
        "cap_reasons": demotion_reasons,
        "info_flags": info_flags,
        "preference_fit": {"salary": salary_fit, "location": location_fit},
        "decision_flags": decision_flags,
    }


# ---------------------------------------------------------------------------
# Pipeline orchestration — shared by every provider
# ---------------------------------------------------------------------------

_RETRY_HINT = "\n\nYour previous response could not be parsed: {error}. Return ONLY a single valid JSON object matching the schema exactly, with no commentary, no markdown fence, and no missing or duplicate IDs."


_MAX_PARSE_ATTEMPTS = 3  # verified live (2026-09-23): Claude mis-escapes free-text
# fields (candidate_evidence quoting the CV) into invalid JSON on ~15% of
# individual CV-matching calls — a single retry left a ~15% chance of a
# job-level failure across a 20-job batch (3/20 observed); at 3 attempts the
# compounded failure probability drops to roughly 0.15^3 ≈ 0.3%.


class ParseFailure(RuntimeError):
    """Raised when every retry attempt failed to parse — carries the last
    raw response text so a failure can be diagnosed without having to
    reproduce it with a fresh, separate API call."""

    def __init__(self, message: str, raw_text: str | None):
        super().__init__(message)
        self.raw_text = raw_text


def _append_retry_hint(prompt: str | CachedPrompt, error: str) -> str | CachedPrompt:
    """Appends to whatever varies between calls, never to the cached
    prefix — mutating the cached half would change it from the previous
    call and guarantee a cache miss on the retry (defeating the whole
    point, on the one call that's now making a second request anyway)."""
    hint = _RETRY_HINT.format(error=error)
    if isinstance(prompt, CachedPrompt):
        return CachedPrompt(prompt.cached, prompt.variable + hint)
    return prompt + hint


def _call_and_parse(
    call: Callable[[str | CachedPrompt], str], prompt: str | CachedPrompt, parse_fn: Callable[[str], dict[str, Any]],
    on_step: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Bounded retries on a parse/schema failure. Catches ValueError
    (includes json.JSONDecodeError, a ValueError subclass, and every
    explicit schema-invalid raise in the extract_*() functions) plus
    AttributeError/TypeError/KeyError — verified live (2026-09-23) that an
    unexpected response SHAPE (e.g. a string where a dict was expected, a
    list item that isn't a dict) throws one of those instead of a clean
    ValueError from our own validation, and previously slipped straight
    past this retry loop on attempt 1 with no retry at all."""
    last_exc: Exception | None = None
    last_raw: str | None = None
    current_prompt = prompt
    for attempt in range(_MAX_PARSE_ATTEMPTS):
        if attempt > 0 and on_step:
            on_step(f"Retrying (attempt {attempt + 1} of {_MAX_PARSE_ATTEMPTS})…")
        raw = call(current_prompt)
        last_raw = raw
        try:
            return parse_fn(raw)
        except (ValueError, AttributeError, TypeError, KeyError) as exc:
            last_exc = exc
            current_prompt = _append_retry_hint(prompt, f"{type(exc).__name__}: {exc}")
    raise ParseFailure(f"{_MAX_PARSE_ATTEMPTS} attempts all failed to parse: {last_exc}", last_raw) from last_exc


_DEDUPE_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "with", "for", "to", "is", "are", "be",
    "ensure", "experience", "strong", "working", "work", "within", "implemented", "correctly",
    "during", "ability", "abilities", "skills", "excellent", "effectively", "both", "including",
    "use", "deployment", "troubleshooting", "management",
}
_DEDUPE_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> set[str]:
    words = _DEDUPE_WORD_RE.findall(text.lower())
    return {w for w in words if w not in _DEDUPE_STOPWORDS and len(w) > 1}


def _remove_eligibility_duplicated_work_conditions(
    requirements: list[dict[str, Any]], eligibility_requirements: list[dict[str, Any]], threshold: float = 0.3,
) -> tuple[list[dict[str, Any]], int]:
    """Drop a work_condition requirement that just restates an eligibility
    item already captured separately (e.g. "Must hold a valid driver's
    license" as BOTH an eligibility_requirement and a work_condition
    requirement) — verified on real data: 7 real cases across both
    providers, e.g. driver's license, work rights, police/criminal-history
    checks each duplicated this way. Not currently a scoring bug
    (work_condition is already excluded from capability_score and CV
    matching — see CAPABILITY_CATEGORIES) but it inflates requirement
    counts and clutters extraction output. Threshold is lower than
    _dedupe_requirements' because eligibility_label and a work_condition
    sentence are worded quite differently even when they're the same
    real-world fact."""
    elig_tokens = [_content_tokens(e.get("eligibility_label") or e.get("eligibility_type") or "") for e in eligibility_requirements]
    elig_tokens = [t for t in elig_tokens if t]
    if not elig_tokens:
        return requirements, 0
    kept = []
    removed = 0
    for req in requirements:
        if req.get("category") != "work_condition":
            kept.append(req)
            continue
        tokens = _content_tokens(req.get("requirement") or "")
        if tokens and any(len(tokens & et) / len(tokens | et) > threshold for et in elig_tokens):
            removed += 1
            continue
        kept.append(req)
    return kept, removed


def _dedupe_requirements(requirements: list[dict[str, Any]], threshold: float = 0.7) -> tuple[list[dict[str, Any]], list[str]]:
    """Collapse near-identical requirements (same capability, reworded) down
    to one entry — keeping whichever occurrence has the MORE SEVERE
    importance (mandatory > core > preferred), not just whichever came
    first. First-wins was the original policy and was wrong: if Pass 1
    extracts something as "preferred" and the audit pass later correctly
    identifies the same requirement as "mandatory" (a real, more careful
    read), first-wins would keep the weaker "preferred" reading and
    permanently underweight a genuine mandatory requirement.

    First attempt (2026-09-23) used whole-string character similarity
    (difflib.SequenceMatcher) and was actively wrong: correctly split
    compound bullets — e.g. "Configuring and troubleshooting networks,
    firewalls, switches, Wi-Fi, VLANs and VPNs" split into 6 separate
    atomic requirements per job_extraction_instructions rule 1 — all share
    boilerplate phrasing ("Configure and troubleshoot X") and the same
    source_quote, so character similarity flagged genuinely distinct items
    (firewalls vs switches vs VLANs) as duplicates at a similar or higher
    score than actual duplicates. Verified on real data: false-positive
    pairs scored up to 0.60 on Jaccard similarity of CONTENT words (with
    generic connector words like "ensure"/"experience"/"strong" removed);
    genuine duplicates (audit re-extracting an existing item, reworded)
    scored 0.44-1.0. Threshold 0.7 sits above every observed false positive
    and still catches near-exact duplicates; it will miss a loosely-worded
    true duplicate occasionally — an acceptable trade since merging a
    distinct requirement into another is the worse failure mode (it
    silently deletes a real capability from scoring)."""
    deduped: list[dict[str, Any]] = []
    conflicts: list[str] = []
    seen: list[tuple[set[str], str]] = []  # (tokens, category), index-aligned with deduped
    for req in requirements:
        tokens = _content_tokens(req.get("requirement") or "")
        category = req.get("category")
        match_idx = None
        for i, (seen_tokens, seen_category) in enumerate(seen):
            if not tokens or not seen_tokens or category != seen_category:
                continue
            if len(tokens & seen_tokens) / len(tokens | seen_tokens) > threshold:
                match_idx = i
                break
        if match_idx is None:
            deduped.append(req)
            seen.append((tokens, category))
            continue

        existing = deduped[match_idx]
        if existing.get("importance") != req.get("importance"):
            existing_severity = IMPORTANCE_WEIGHT.get(existing.get("importance"), 1)
            new_severity = IMPORTANCE_WEIGHT.get(req.get("importance"), 1)
            winner = req if new_severity > existing_severity else existing
            conflicts.append(
                f"{existing.get('requirement_id')} ({existing.get('importance')!r}) vs "
                f"{req.get('requirement_id')} ({req.get('importance')!r}) — kept the more severe: {winner.get('importance')!r}"
            )
            if new_severity > existing_severity:
                deduped[match_idx] = req
                seen[match_idx] = (tokens, category)
        # else: identical importance, keep the existing (first) one as-is.
    return deduped, conflicts


def _renumber_requirements(requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign final requirement_ids in Python, not trusted from the model —
    Pass 1 and the audit pass number independently (the audit is only told
    to "continue from R{n}"), so post-merge/dedup IDs could otherwise
    collide or skip. Renumbering happens before Pass 3 is built, so
    CV-matching naturally uses the final IDs with no separate remap step."""
    for i, req in enumerate(requirements, 1):
        req["requirement_id"] = f"R{i:02d}"
    return requirements


def hard_disqualified_result(job_extraction: dict[str, Any], calls_made: int) -> dict[str, Any] | None:
    """Eligibility and industry exclusion are both decided entirely from
    Pass-1 fields (eligibility_requirements, employer_industry) against
    profile config — no LLM call needed, and the audit pass never adds to
    either field (it only appends capability requirements). When either
    one hard-disqualifies the job, the recommendation is already decided
    regardless of how well the CV matches, so the audit and CV-matching
    passes — the two most expensive calls — would be pure cost with zero
    effect on the outcome. Returns the finished result, or None if the job
    must go on to the remaining passes."""
    prefs = _load_prefs()
    eligibility = compute_eligibility(job_extraction.get("eligibility_requirements") or [], prefs.get("eligibility", {}))
    unmet_eligibility = [k for k, v in eligibility.items() if v == "unmet"]
    industry_status, industry_label = compute_industry_fit(job_extraction.get("employer_industry"), prefs)
    if not (unmet_eligibility or industry_status == "excluded"):
        return None
    recommendation = "Ineligible" if unmet_eligibility else "Excluded"
    cap_reasons = (
        [f"unmet eligibility requirement ({unmet_eligibility[0]}) — job is not applicable regardless of skill fit"]
        if unmet_eligibility
        else [f"excluded industry: {industry_label} — job matches an employer type you've asked to exclude, regardless of role fit"]
    )
    job_extraction["extraction_stats"] = {
        "pass1_count": len(job_extraction.get("requirements") or []),
        "skipped_audit_and_cv_matching": True,
        "skip_reason": cap_reasons[0],
    }
    return {
        "cv_fit_score": None,
        "capability_score": None,
        "salary_fit": None,
        "location_fit": None,
        "industry_fit": {"status": industry_status, "label": industry_label},
        "eligibility": eligibility,
        "overall_score": None,
        "recommendation": recommendation,
        "pitch_angle": "",
        "gaps": [],
        "cap_reasons": cap_reasons,
        "info_flags": ["skipped CV matching — job already disqualified by eligibility or excluded industry"],
        "preference_fit": {"salary": None, "location": None},
        "decision_flags": {},
        "calls_made": calls_made,
        "job_extraction": job_extraction,
        "cv_matching": {},
    }


def merge_audit_and_dedupe(job_extraction: dict[str, Any], audit: dict[str, Any] | None) -> set[str]:
    """Folds the (optional) audit findings into the Pass-1 requirements,
    dedupes/renumbers them, records extraction_stats, and returns the ids
    CV-matching must answer for. Shared by run_pipeline and the batch runner."""
    pre_audit_count = len(job_extraction.get("requirements") or [])
    audit_added_count = 0
    audit_fuzzy_quotes: list[str] = []
    if audit is not None:
        audit_added_count = len(audit["missing_requirements"])
        audit_fuzzy_quotes = audit.get("_fuzzy_quotes", [])
        job_extraction["requirements"] = (job_extraction.get("requirements") or []) + audit["missing_requirements"]

    fuzzy_quotes = job_extraction.pop("_fuzzy_quotes", []) + audit_fuzzy_quotes
    requirements_no_elig_dupes, eligibility_duplicates_removed = _remove_eligibility_duplicated_work_conditions(
        job_extraction.get("requirements") or [], job_extraction.get("eligibility_requirements") or []
    )
    deduped, importance_conflicts = _dedupe_requirements(requirements_no_elig_dupes)
    job_extraction["requirements"] = _renumber_requirements(deduped)
    final_count = len(job_extraction["requirements"])
    job_extraction["extraction_stats"] = {
        "pass1_count": pre_audit_count,
        "audit_added": audit_added_count,
        "eligibility_duplicates_removed": eligibility_duplicates_removed,
        "semantic_duplicates_removed": len(requirements_no_elig_dupes) - final_count,
        "final_count": final_count,
        "importance_conflicts": importance_conflicts,
        "extraction_fuzzy_quotes": fuzzy_quotes,
    }
    return {r["requirement_id"] for r in job_extraction["requirements"] if r["category"] in CAPABILITY_CATEGORIES}


def finish_scoring(job_extraction: dict[str, Any], cv_matching: dict[str, Any], calls_made: int) -> dict[str, Any]:
    job_extraction["extraction_stats"]["cv_evidence_fuzzy_quotes"] = cv_matching.pop("_fuzzy_evidence", [])
    result = compute_final_result(job_extraction, cv_matching)
    result["calls_made"] = calls_made
    result["job_extraction"] = job_extraction
    result["cv_matching"] = cv_matching
    return result


def run_pipeline(
    call: Callable[[str | CachedPrompt], str], title: str, company: str, description: str, force_audit: bool = False,
    on_step: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Runs the full job-extraction → conditional-audit → CV-matching →
    deterministic-scoring pipeline against any provider. `call` is a plain
    function of (prompt: str | CachedPrompt) -> str — the only thing that
    differs between providers. job_extraction/cv_matching prompts are a
    CachedPrompt (cacheable prefix + variable suffix, see CachedPrompt);
    the audit prompt is a plain str (too small to benefit from caching).
    Both score_job() (Anthropic, production) and any comparison script
    (e.g. against DeepSeek) call this same function, so a difference
    in results can only come from extraction quality, never from different
    scoring logic between providers.

    force_audit=True always runs the audit pass regardless of needs_audit()
    — intended for benchmark/comparison runs where extraction completeness
    itself is under test, not for production (where the conditional
    heuristic keeps cost down).

    on_step, if given, is called with a short human-readable label at each
    stage transition — purely cosmetic (e.g. a UI polling a manual-rescore
    button's progress), never affects scoring."""
    if on_step:
        on_step("Reading the job ad…")
    job_extraction = _call_and_parse(
        call, build_job_extraction_prompt(title, company, description),
        lambda text: extract_job_requirements(text, title, company, description),
        on_step,
    )
    calls_made = 1

    # Eligibility and industry exclusion are both decided entirely from
    # Pass-1 fields (eligibility_requirements, employer_industry) against
    # profile config — no LLM call needed, and the audit pass never adds to
    # either field (it only appends capability requirements). When either
    # one hard-disqualifies the job, the recommendation is already decided
    # regardless of how well the CV matches, so the audit and CV-matching
    # passes — the two most expensive calls — would be pure cost with zero
    # effect on the outcome. Short-circuit here instead of running them.
    if not force_audit:
        disqualified = hard_disqualified_result(job_extraction, calls_made)
        if disqualified is not None:
            if on_step:
                on_step("Calculating score…")
            return disqualified

    audit = None
    if force_audit or needs_audit(job_extraction, description):
        if on_step:
            on_step("Double-checking for missed requirements…")
        audit = _call_and_parse(
            call, build_audit_prompt(title, company, description, job_extraction),
            lambda text: extract_audit_findings(text, title, company, description),
            on_step,
        )
        calls_made += 1

    expected_ids = merge_audit_and_dedupe(job_extraction, audit)

    if on_step:
        on_step("Matching against your CV…")
    cv_matching = _call_and_parse(
        call, build_cv_matching_prompt(job_extraction),
        lambda text: extract_cv_matches(text, expected_ids),
        on_step,
    )
    calls_made += 1

    if on_step:
        on_step("Calculating score…")
    return finish_scoring(job_extraction, cv_matching, calls_made)


def _thinking_off_param(model: str) -> dict[str, str]:
    """Every current Claude model runs adaptive (hidden-reasoning) thinking
    by default when `thinking` is omitted — not "no thinking", like older
    models. Verified live (2026-10-02) against real job_extraction and
    cv_matching calls: adaptive thinking added 43-64% MORE output tokens
    for an IDENTICAL-length JSON result (same schema, same content) — pure
    overhead on a deterministic extraction/classification task that gains
    nothing from chain-of-thought. Output tokens are priced 5x input
    tokens, so this was the single largest line item nobody had measured,
    since `response.usage` was never logged anywhere in this pipeline.
    Sonnet 5 accepts {"type": "disabled"} directly; Sonnet 5.5 rejects that
    (400) and needs {"type": "between_tools"} instead (only valid at
    effort "high" or below, which is this pipeline's unset/default)."""
    if model == "claude-sonnet-5-5":
        return {"type": "between_tools"}
    return {"type": "disabled"}


def _anthropic_call(client: Anthropic, prompt: str | CachedPrompt, max_tokens: int = 8192) -> str:
    """A CachedPrompt's `cached` half is sent as its own content block with
    a cache_control breakpoint — Anthropic caches everything up to and
    including that block (ephemeral, ~5 min TTL) and, on a later call with
    a byte-identical prefix, charges ~10% of normal input price for it
    instead of re-billing it in full. `variable` is a second, uncached
    block appended after. A plain str prompt (the audit pass) is sent as
    before, uncached."""
    if isinstance(prompt, CachedPrompt):
        content = [
            {"type": "text", "text": prompt.cached, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": prompt.variable},
        ]
    else:
        content = prompt
    response = client.messages.create(
        model=MODEL, max_tokens=max_tokens, thinking=_thinking_off_param(MODEL),
        metadata={"user_id": f"jobsearch-{PROFILE}"},  # lets the Console split spend per profile
        messages=[{"role": "user", "content": content}],
    )
    _log_usage(response.usage)
    return "".join(b.text for b in response.content if b.type == "text")


def _log_usage(usage: Any, batch: bool = False) -> None:
    """One JSON line per API call in the profile dir — Anthropic's billing
    ledger isn't reachable from here without an Admin key, and without this
    there was no way to confirm caching/thinking changes actually moved the
    bill. Never lets a logging failure break scoring."""
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "input": usage.input_tokens,
            "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "output": usage.output_tokens,
            "batch": batch,  # billed at 50% - cost_report.py applies the discount
        }
        with (PROFILE_DIR / "api_usage.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def score_job(
    title: str, company: str, description: str, on_step: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Score one alert preview via Claude — the production entrypoint used
    by app.py/ingest.py. Return shape is unchanged from earlier designs
    (cv_fit_score, salary_fit, location_fit, overall_score, recommendation,
    pitch_angle, gaps) so those callers don't need to change; eligibility is
    new (a dict of type -> status) but purely additive."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    client = Anthropic()
    return run_pipeline(lambda prompt: _anthropic_call(client, prompt), title, company, description, on_step=on_step)
