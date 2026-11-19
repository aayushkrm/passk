"""The headline contract: schema, selection rules, formatting and README consistency.

Light by design: fixture reports and the committed calibration only. No model is run.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from passk.diff import load_report
from passk.schema import RunReport

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
SCHEMA = json.loads((REPO / "schemas" / "headline.schema.json").read_text(encoding="utf-8"))
CALIBRATION = json.loads((REPO / "results" / "calibration" / "validate.json").read_text())


def _render_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "render_readme", REPO / "scripts" / "render_readme.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RR = _render_module()


# A small validator for the subset of JSON Schema that headline.schema.json uses. It keeps CI
# free of an extra dependency.
def _check(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    if "$ref" in schema:
        target = SCHEMA
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        return _check(instance, target, path)
    errors: list[str] = []
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        ok = any(_is_type(instance, kind) for kind in kinds)
        if not ok:
            return [f"{path}: expected type {kinds}, got {type(instance).__name__}"]
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: does not match {schema['pattern']}")
    if isinstance(instance, int) and not isinstance(instance, bool) and "minimum" in schema:
        if instance < schema["minimum"]:
            errors.append(f"{path}: below {schema['minimum']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing {key}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in props:
                    errors.append(f"{path}: unexpected {key}")
        for key, sub in props.items():
            if key in instance:
                errors += _check(instance[key], sub, f"{path}.{key}")
    if isinstance(instance, list):
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        for index, element in enumerate(instance):
            if "items" in schema:
                errors += _check(element, schema["items"], f"{path}[{index}]")
    return errors


def _is_type(value: Any, kind: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "null": value is None,
    }[kind]


def _validate(instance: Any) -> list[str]:
    return _check(instance, SCHEMA)


def _report(path: str, run_id: str, *, complete: bool = True, repo: str | None = None) -> RunReport:
    base = load_report(FIXTURES / path)
    env = dict(base.env_info)
    if repo:
        env["agent"] = {"kind": "llamacpp", "repo_id": repo, "model_file": "x.gguf"}
        env.pop("synthetic", None)  # a model run is not synthetic
    update: dict[str, Any] = {
        "run_id": run_id,
        "env_info": env,
        "config": base.config.model_copy(update={"run_id": run_id}),
    }
    if not complete:
        update.update(
            complete=False,
            stop_reason="provider returned HTTP 429: account quota exhausted",
            reset_at="2026-11-20T06:00:00+00:00",
            episodes_done=6,
            episodes_total=30,
        )
    return base.model_copy(update=update)


def _results(reports: list[RunReport], calibration: dict[str, Any] | None) -> dict[str, Any]:
    paths = [(Path(RR.RESULTS / "e3" / r.run_id / "report.json"), r) for r in reports]
    return {"reports": paths, "runs": [], "calibration": calibration, "gates": []}


def test_empty_results_give_an_empty_but_valid_contract() -> None:
    contract = RR.build_headline(_results([], None))
    assert contract["items"] == []
    assert _validate(contract) == []


def test_committed_headline_is_valid_and_matches_the_generator() -> None:
    committed = json.loads((REPO / "results" / "headline.json").read_text(encoding="utf-8"))
    assert _validate(committed) == []
    assert committed == RR.build_headline()


def test_standard_keys_and_links_are_exact() -> None:
    contract = RR.build_headline(_results([], None))
    assert contract["schema_version"] == "1"
    assert contract["tool"] == "passk"
    assert contract["links"] == {
        "repo": "https://github.com/aayushkrm/passk",
        "site": "https://aayushkrm.github.io/passk/",
        "release": None,
    }


def test_items_are_capped_and_follow_the_priority_order() -> None:
    reports = [
        _report(
            "report_baseline.json", "e3-qwen2.5-0.5b-t0.7", repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        ),
        _report(
            "report_baseline.json", "e3-qwen2.5-1.5b-t0.7", repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF"
        ),
        _report("report_baseline.json", "e5-qwen2.5-1.5b-v2-t0.7"),
        _report("report_minor_drop.json", "e7-nemotron-t0.7", complete=False),
        _report("report_baseline.json", "e3-qwen3-1.7b-q8-t0.7", repo="Qwen/Qwen3-1.7B-GGUF"),
    ]
    results = _results(reports, CALIBRATION)
    items = RR.build_headline(results)["items"]
    assert len(items) == 6
    assert [item["id"] for item in items] == [
        "e3-qwen2.5-0.5b",
        "e3-qwen2.5-1.5b",
        "e5-regression",
        "e7-hosted",
        "e1-estimator",
        "e2-gate",
    ]
    assert "e3-qwen3-1.7b" not in [item["id"] for item in items]  # optional arm, lowest priority


def test_items_are_omitted_when_their_source_is_missing() -> None:
    items = RR.build_headline(_results([], CALIBRATION))["items"]
    assert [item["id"] for item in items] == ["e1-estimator", "e2-gate"]
    assert RR.build_headline(_results([], None))["items"] == []


def test_calibration_items_are_synthetic_and_point_at_their_json() -> None:
    items = {i["id"]: i for i in RR.build_headline(_results([], CALIBRATION))["items"]}
    for item_id in ("e1-estimator", "e2-gate"):
        assert items[item_id]["synthetic"] is True
        assert items[item_id]["featured"] is False
        assert items[item_id]["source"] == "results/calibration/validate.json"
    assert items["e1-estimator"]["value"].startswith("max abs error ")
    assert items["e2-gate"]["value"].startswith("false alarm ")


def test_e3_item_is_featured_gap_with_a_ci_and_no_invented_n() -> None:
    report = _report(
        "report_baseline.json", "e3-qwen2.5-0.5b-t0.7", repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    )
    item = RR.build_headline(_results([report], None))["items"][0]
    assert item["id"] == "e3-qwen2.5-0.5b"
    assert item["featured"] is True and item["synthetic"] is False
    assert re.fullmatch(r"pass@1 \d\.\d{3} vs pass\^5 \d\.\d{3}, gap -?\d\.\d{3}", item["value"])
    assert re.fullmatch(r"95% CI on the gap \[-?\d\.\d{3}, -?\d\.\d{3}\]", item["ci"])
    assert item["n"] == sum(t.n for t in report.tasks)
    assert "Qwen2.5-0.5B-Instruct" in item["detail"]


def test_failure_leader_excludes_success_and_uses_episode_share() -> None:
    report = _report("report_baseline.json", "e3-qwen2.5-0.5b-t0.7")
    leader = RR._leader(report)
    totals = {k: v for k, v in report.failure_totals.items() if k != "success"}
    name = max(sorted(totals.items()), key=lambda kv: kv[1])[0]
    assert leader.startswith(name + " ")
    assert leader.endswith(" of episodes")
    assert "success" not in leader


def test_partial_e3_run_is_labelled_partial_in_its_value() -> None:
    report = _report("report_baseline.json", "e3-qwen2.5-1.5b-t0.7", complete=False)
    item = RR.build_headline(_results([report], None))["items"][0]
    assert item["value"].startswith("PARTIAL: ")
    assert "Stopped early" in item["detail"]


def test_e5_verdict_is_a_gate_decision_and_not_evaluated_for_partial_runs() -> None:
    baseline = _report("report_baseline.json", "e3-qwen2.5-1.5b-t0.7")
    regression = _report("report_regression.json", "e5-qwen2.5-1.5b-v2-t0.7")
    item = next(
        i
        for i in RR.build_headline(_results([baseline, regression], None))["items"]
        if i["id"] == "e5-regression"
    )
    assert item["value"] in {"PASS", "FAIL"}
    assert item["value"] == "FAIL"  # report_regression is the planted regression fixture
    assert item["source"].startswith("results/e3/") or item["source"].startswith("results/")
    partial = _report("report_regression.json", "e5-qwen2.5-1.5b-v2-t0.7", complete=False)
    item = next(
        i
        for i in RR.build_headline(_results([baseline, partial], None))["items"]
        if i["id"] == "e5-regression"
    )
    assert item["value"] == "not evaluated: partial run"


def test_e7_partial_is_reported_with_progress_and_reset_time() -> None:
    report = _report("report_minor_drop.json", "e7-nemotron-t0.7", complete=False)
    item = next(
        i for i in RR.build_headline(_results([report], None))["items"] if i["id"] == "e7-hosted"
    )
    assert item["value"] == "PARTIAL: 6 of 30 episodes"
    assert "2026-11-20T06:00:00+00:00" in item["detail"]
    assert item["n"] == 6 and item["featured"] is False


def test_readme_results_block_shows_every_headline_value() -> None:
    results = _results([_report("report_baseline.json", "e3-qwen2.5-0.5b-t0.7")], CALIBRATION)
    contract = RR.build_headline(results)
    block = RR._block(results, contract)
    for item in contract["items"]:
        assert item["value"] in block, item["id"]
        assert item["label"] in block, item["id"]


def test_headline_sources_are_relative_and_present() -> None:
    contract = RR.build_headline()
    for item in contract["items"]:
        assert not item["source"].startswith("/")
        assert (REPO / item["source"]).is_file(), item["source"]
        assert "/Users/" not in json.dumps(item)


def test_item_details_say_what_n_counts() -> None:
    report = _report(
        "report_baseline.json", "e3-qwen2.5-0.5b-t0.7", repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    )
    item = RR.build_headline(_results([report], CALIBRATION))["items"][0]
    assert "n counts episodes" in item["detail"]
    e2 = next(
        i for i in RR.build_headline(_results([], CALIBRATION))["items"] if i["id"] == "e2-gate"
    )
    assert "n counts the pairs" in e2["detail"]
