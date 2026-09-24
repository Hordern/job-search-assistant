"""Lightweight regression tests for the requirement dedup/renumber/eligibility
logic in scorer.py — plain assertions, no pytest dependency. The scoring code
has changed several times in one session; these exist so the next edit gets
caught by running this file, not by re-eyeballing a fresh CSV.

Usage:
    JOBSEARCH_PROFILE=alice python test_scorer.py
"""

import json

import scorer


def req(rid, text, importance="core", category="skill", quote=None):
    return {
        "requirement_id": rid, "requirement": text, "importance": importance, "category": category,
        "source_quote": quote or text, "source_section": "", "years_required": None,
    }


def test_same_requirement_worded_differently():
    reqs = [
        req("R01", "Work within an MSP or IT services environment"),
        req("R02", "Experience working within an MSP or IT services environment"),
    ]
    deduped, conflicts = scorer._dedupe_requirements(reqs)
    assert len(deduped) == 1, f"expected the reworded restatement to be merged, got {len(deduped)}"
    assert deduped[0]["requirement_id"] == "R01", "must keep the FIRST occurrence, not the second"
    assert conflicts == []


def test_distinct_technologies_sharing_one_source_sentence():
    quote = "Configuring and troubleshooting networks, firewalls, switches, Wi-Fi, VLANs and VPNs."
    reqs = [
        req("R01", "Configure and troubleshoot networks", quote=quote),
        req("R02", "Configure and troubleshoot firewalls", quote=quote),
        req("R03", "Configure and troubleshoot switches", quote=quote),
        req("R04", "Configure and troubleshoot VLANs", quote=quote),
        req("R05", "Configure and troubleshoot VPNs", quote=quote),
    ]
    deduped, conflicts = scorer._dedupe_requirements(reqs)
    assert len(deduped) == 5, (
        f"distinct technologies sharing boilerplate phrasing AND the same source_quote must NOT be merged "
        f"(this was the original bug) — got {len(deduped)}"
    )


def test_mandatory_and_preferred_versions_keeps_more_severe():
    # more-severe (mandatory) appears FIRST — trivially kept
    reqs = [
        req("R01", "5+ years SharePoint Online migration experience", importance="mandatory"),
        req("R02", "SharePoint Online migration experience", importance="preferred"),
    ]
    deduped, conflicts = scorer._dedupe_requirements(reqs)
    assert len(deduped) == 1
    assert deduped[0]["importance"] == "mandatory"
    assert len(conflicts) == 1, "an importance mismatch between merged items must be recorded, not silently dropped"

    # more-severe (mandatory) appears SECOND — must still win, not "whichever came first"
    reqs2 = [
        req("R01", "SharePoint Online migration experience", importance="preferred"),
        req("R02", "5+ years SharePoint Online migration experience", importance="mandatory"),
    ]
    deduped2, conflicts2 = scorer._dedupe_requirements(reqs2)
    assert len(deduped2) == 1
    assert deduped2[0]["importance"] == "mandatory", (
        "a more severe duplicate arriving SECOND (e.g. from the audit pass) must still win over a weaker first read"
    )
    assert len(conflicts2) == 1


def test_eligibility_duplicated_as_work_condition():
    requirements = [
        req("R01", "Must hold a valid driver's license with own transport", category="work_condition"),
        req("R02", "Strong troubleshooting skills", category="skill"),
    ]
    eligibility_requirements = [{
        "eligibility_id": "E01", "eligibility_type": "drivers_license",
        "eligibility_label": "Valid driver's license with own transport",
        "timing": "must_already_hold", "mandatory": True, "source_quote": "valid driver's license",
    }]
    cleaned, removed_count = scorer._remove_eligibility_duplicated_work_conditions(requirements, eligibility_requirements)
    ids = {r["requirement_id"] for r in cleaned}
    assert ids == {"R02"}, f"the work_condition restating the eligibility item should be dropped, got {ids}"
    assert removed_count == 1


def test_audit_adding_already_represented_requirement():
    pass1 = [req("R01", "Strong customer relationship and stakeholder management skills")]
    audit_additions = [req("R05", "Strong customer relationship and stakeholder management skills")]
    combined = pass1 + audit_additions
    deduped, conflicts = scorer._dedupe_requirements(combined)
    assert len(deduped) == 1, "an audit addition that's byte-identical to an existing requirement must be dropped"
    assert deduped[0]["requirement_id"] == "R01"


def test_renumbering_produces_sequential_ids_and_pass3_uses_them():
    reqs = [req("R01", "A"), req("R05", "B"), req("R09", "C")]  # gappy IDs, as a merge could produce
    renumbered = scorer._renumber_requirements(reqs)
    ids = [r["requirement_id"] for r in renumbered]
    assert ids == ["R01", "R02", "R03"], f"renumbering must be sequential from R01, got {ids}"

    job_extraction = {
        "requirements": renumbered, "role_level_stated": "mid", "role_family": "IT Support",
    }
    prompt = scorer.build_cv_matching_prompt(job_extraction)
    for rid in ids:
        assert rid in prompt, f"renumbered id {rid} must appear in the Pass 3 prompt built AFTER renumbering"
    assert "R05" not in prompt and "R09" not in prompt, "stale pre-renumber ids must not leak into the Pass 3 prompt"


def test_ineligible_only_fires_on_mandatory_unheld_unwilling():
    # mandatory + not held + not willing -> unmet -> Ineligible
    elig = scorer.compute_eligibility(
        [{"eligibility_type": "drivers_license", "timing": "must_already_hold", "mandatory": True}],
        {"drivers_license": {"has": False, "willing_to_obtain": False}},
    )
    assert elig["drivers_license"] == "unmet"

    # willing -> never unmet, regardless of timing
    elig = scorer.compute_eligibility(
        [{"eligibility_type": "drivers_license", "timing": "must_already_hold", "mandatory": True}],
        {"drivers_license": {"has": False, "willing_to_obtain": True}},
    )
    assert elig["drivers_license"] != "unmet", "willing_to_obtain must never produce unmet"

    # not configured -> unknown, never unmet
    elig = scorer.compute_eligibility(
        [{"eligibility_type": "security_clearance", "timing": "must_already_hold", "mandatory": True}], {},
    )
    assert elig["security_clearance"] == "unknown"

    # not mandatory -> never unmet even if not held/not willing
    elig = scorer.compute_eligibility(
        [{"eligibility_type": "police_check", "timing": "must_already_hold", "mandatory": False}],
        {"police_check": {"has": False, "willing_to_obtain": False}},
    )
    assert elig["police_check"] != "unmet", "a non-mandatory item must never produce unmet"


def test_duplicate_eligibility_type_aggregates_to_strictest():
    # extracted twice: first weak (preferred, can-obtain-later), second strict (mandatory, must-already-hold)
    # last-wins would use the SECOND (mandatory/must_already_hold) by accident of order — must be right either way,
    # so also test the REVERSE order to prove it's aggregation, not luck.
    for order in (
        [{"eligibility_type": "drivers_license", "timing": "can_obtain_after_appointment", "mandatory": False},
         {"eligibility_type": "drivers_license", "timing": "must_already_hold", "mandatory": True}],
        [{"eligibility_type": "drivers_license", "timing": "must_already_hold", "mandatory": True},
         {"eligibility_type": "drivers_license", "timing": "can_obtain_after_appointment", "mandatory": False}],
    ):
        elig = scorer.compute_eligibility(order, {"drivers_license": {"has": False, "willing_to_obtain": False}})
        assert elig["drivers_license"] == "unmet", (
            f"aggregation must keep the strictest combination (mandatory + must_already_hold) regardless of "
            f"extraction order, got {elig['drivers_license']!r}"
        )


def test_cv_matches_rejects_missing_and_invented_ids():
    expected = {"R01", "R02", "R03"}
    # must be real CV text — a made-up string would fail quote validation
    # first and mask the ID-coverage check this test targets.
    real_cv_snippet = scorer._load_cv()[200:260].strip()

    def match(rid, state="direct", depth="production", evidence=None):
        return {"requirement_id": rid, "match_state": state, "evidence_depth": depth, "candidate_evidence": evidence or real_cv_snippet}

    missing_payload = json.dumps({
        "matches": [match("R01"), match("R02")],  # R03 missing
        "scope_match": "aligned", "confidence": "high", "pitch_angle": "x", "gaps": [],
    })
    try:
        scorer.extract_cv_matches(missing_payload, expected)
        assert False, "must reject a response missing a requirement_id"
    except ValueError as e:
        assert "missing" in str(e).lower()

    invented_payload = json.dumps({
        "matches": [match("R01"), match("R02"), match("R03"), match("R99")],
        "scope_match": "aligned", "confidence": "high", "pitch_angle": "x", "gaps": [],
    })
    try:
        scorer.extract_cv_matches(invented_payload, expected)
        assert False, "must reject a response with an invented requirement_id"
    except ValueError as e:
        assert "invented" in str(e).lower() or "unexpected" in str(e).lower()


def test_cv_matches_rejects_depth_state_mismatch_and_missing_evidence():
    expected = {"R01"}

    def payload(match_dict):
        return json.dumps({
            "matches": [{"requirement_id": "R01", **match_dict}],
            "scope_match": "aligned", "confidence": "high", "pitch_angle": "x", "gaps": [],
        })

    # direct + inferred depth: not allowed, direct must be depth=production
    try:
        scorer.extract_cv_matches(
            payload({"match_state": "direct", "evidence_depth": "inferred", "candidate_evidence": "something"}), expected,
        )
        assert False, "direct + evidence_depth=inferred must be rejected"
    except ValueError:
        pass

    # adjacent with no evidence: not allowed, adjacent needs real evidence too
    try:
        scorer.extract_cv_matches(
            payload({"match_state": "adjacent", "evidence_depth": "inferred", "candidate_evidence": None}), expected,
        )
        assert False, "adjacent with no candidate_evidence must be rejected"
    except ValueError:
        pass

    # not_evidenced with non-null evidence: not allowed
    try:
        scorer.extract_cv_matches(
            payload({"match_state": "not_evidenced", "evidence_depth": "none", "candidate_evidence": "invented text"}), expected,
        )
        assert False, "not_evidenced must have null candidate_evidence"
    except ValueError:
        pass


def test_unknown_preferences_do_not_demote_recommendation():
    job_extraction = {
        "requirements": [req("R01", "Python scripting", importance="core")],
        "eligibility_requirements": [], "salary_min": None, "salary_max": None,
        "location_text": None, "work_arrangement": "unclear", "onsite_days": None,
    }
    cv_matching = {
        "matches": [{"requirement_id": "R01", "match_state": "direct", "evidence_depth": "production", "candidate_evidence": "x"}],
        "scope_match": "aligned",
    }
    result = scorer.compute_final_result(job_extraction, cv_matching)
    assert result["salary_fit"] == "unknown" and result["location_fit"] == "unknown"
    assert not any("unknown salary" in r or "unknown location" in r for r in result["cap_reasons"]), (
        "unknown salary/location must never appear as a demotion reason"
    )
    assert any("not stated" in f for f in result["info_flags"]), "unknown salary/location should still be visible as an info flag"


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    print()
    print(f"{passed}/{len(tests)} passed" + (f", {failed} FAILED" if failed else ""))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
