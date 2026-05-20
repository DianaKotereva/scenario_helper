#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class CheckResult:
    check_id: str
    passed: bool
    message: str
    details: Dict[str, Any]


def norm(s: str) -> str:
    return (s or "").strip().lower().replace("ё", "е")


def contains(hay: str, needle: str) -> bool:
    return norm(needle) in norm(hay)


def load_rules(path: Path) -> Dict[str, Any]:
    # YAML subset parser via PyYAML (already in project deps usually).
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from e

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_batch_nodes(batch_file: Path) -> List[Dict[str, Any]]:
    data = json.loads(batch_file.read_text(encoding="utf-8"))

    nodes: List[Dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if "main_name" in x and "classification" in x:
                nodes.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return nodes


def get_alt_names(node: Dict[str, Any]) -> List[str]:
    alts = node.get("alt_names", [])
    if alts is None:
        return []
    if isinstance(alts, list):
        return [str(a) for a in alts]
    return [str(alts)]


def find_nodes(nodes: List[Dict[str, Any]], main_name: str, exact: bool = False) -> List[Dict[str, Any]]:
    out = []
    for n in nodes:
        m = str(n.get("main_name", ""))
        ok = norm(m) == norm(main_name) if exact else contains(m, main_name)
        if ok:
            out.append(n)
    return out


def check_forbid_alt_on_main(check: Dict[str, Any], nodes: List[Dict[str, Any]], exact_main: bool) -> CheckResult:
    cid = check["id"]
    main = check["main_name"]
    bad_alt = check["forbidden_alt"]

    matched_nodes = find_nodes(nodes, main, exact=exact_main)
    offenders = []
    for n in matched_nodes:
        alts = get_alt_names(n)
        if any(contains(a, bad_alt) for a in alts):
            offenders.append({"main_name": n.get("main_name"), "alt_names": alts})

    if offenders:
        return CheckResult(
            cid,
            False,
            f"Найден запрещённый alt_name '{bad_alt}' у main_name '{main}'",
            {"offenders": offenders},
        )

    return CheckResult(
        cid,
        True,
        "OK",
        {"matched_nodes": len(matched_nodes)},
    )


def check_require_main_name(check: Dict[str, Any], nodes: List[Dict[str, Any]]) -> CheckResult:
    cid = check["id"]
    req = check["required_main_name"]
    matched = find_nodes(nodes, req, exact=False)
    if not matched:
        return CheckResult(cid, False, f"Не найден required main_name '{req}'", {})
    return CheckResult(cid, True, "OK", {"count": len(matched)})


def check_forbid_laurefinde_lizard_like_node(check: Dict[str, Any], nodes: List[Dict[str, Any]]) -> CheckResult:
    cid = check["id"]
    offenders = []
    for n in nodes:
        main = str(n.get("main_name", ""))
        alts = get_alt_names(n)
        text = " | ".join([main] + alts)
        is_lizard_like = (contains(text, "лаур") and contains(text, "ящер"))
        if is_lizard_like:
            offenders.append({"main_name": main, "alt_names": alts})

    if offenders:
        return CheckResult(cid, False, "Найдена нода Лаурэфиндэ (ящер) или подобная", {"offenders": offenders})
    return CheckResult(cid, True, "OK", {})


def check_conditional_forbid_alt_on_main_exact(check: Dict[str, Any], nodes: List[Dict[str, Any]]) -> CheckResult:
    # Same as forbid_alt_on_main_exact but explicit conditional semantics in message
    res = check_forbid_alt_on_main(check, nodes, exact_main=True)
    return res


def check_bidirectional_no_alias_link(check: Dict[str, Any], nodes: List[Dict[str, Any]]) -> CheckResult:
    cid = check["id"]
    pivot = check["pivot_main_name"]
    others = check["unrelated_names"]

    pivot_nodes = find_nodes(nodes, pivot, exact=True)
    if not pivot_nodes:
        return CheckResult(cid, False, f"Не найдена pivot-нода '{pivot}'", {})

    offenders = []

    # 1) pivot must not contain others in alt_names
    for pn in pivot_nodes:
        alts = get_alt_names(pn)
        for o in others:
            if any(contains(a, o) for a in alts):
                offenders.append({
                    "rule": "pivot_has_forbidden_alt",
                    "pivot_main": pn.get("main_name"),
                    "forbidden_name": o,
                    "pivot_alts": alts,
                })

    # 2) each other node must not contain pivot in alt_names
    for o in others:
        other_nodes = find_nodes(nodes, o, exact=True)
        for on in other_nodes:
            alts = get_alt_names(on)
            if any(contains(a, pivot) for a in alts):
                offenders.append({
                    "rule": "other_has_pivot_alt",
                    "other_main": on.get("main_name"),
                    "pivot": pivot,
                    "other_alts": alts,
                })

    if offenders:
        return CheckResult(cid, False, "Найдена запрещённая alias-связь pivot<->others", {"offenders": offenders})

    return CheckResult(cid, True, "OK", {"pivot_nodes": len(pivot_nodes)})


def run_checks(batch_payloads_dir: Path, rules: Dict[str, Any]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    check_defs = rules.get("checks", [])
    for check in check_defs:
        batch = check["batch"]
        batch_file = batch_payloads_dir / f"{batch}.json"

        if not batch_file.exists():
            r = CheckResult(check["id"], False, f"Не найден файл батча: {batch_file.name}", {"batch_file": str(batch_file)})
            results.append(r.__dict__)
            continue

        nodes = load_batch_nodes(batch_file)
        ctype = check["type"]

        if ctype == "forbid_alt_on_main":
            r = check_forbid_alt_on_main(check, nodes, exact_main=False)
        elif ctype == "forbid_alt_on_main_exact":
            r = check_forbid_alt_on_main(check, nodes, exact_main=True)
        elif ctype == "require_main_name":
            r = check_require_main_name(check, nodes)
        elif ctype == "forbid_laurefinde_lizard_like_node":
            r = check_forbid_laurefinde_lizard_like_node(check, nodes)
        elif ctype == "conditional_forbid_alt_on_main_exact":
            r = check_conditional_forbid_alt_on_main_exact(check, nodes)
        elif ctype == "bidirectional_no_alias_link":
            r = check_bidirectional_no_alias_link(check, nodes)
        else:
            r = CheckResult(check["id"], False, f"Неизвестный тип проверки: {ctype}", {})

        out = r.__dict__
        out["batch_file"] = str(batch_file)
        out["check_note"] = check.get("note", "")
        results.append(out)

    passed = sum(1 for x in results if x["passed"])
    failed = len(results) - passed

    return {
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run extraction prompt regression checks.")
    parser.add_argument(
        "--batch-payloads-dir",
        required=True,
        help="Path to results/_batch_payloads",
    )
    parser.add_argument(
        "--rules",
        default="scenatio_helper/validation/extraction_prompt_checks.yaml",
        help="Path to YAML rules file",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to JSON report file",
    )
    args = parser.parse_args()

    batch_dir = Path(args.batch_payloads_dir)
    rules_path = Path(args.rules)

    rules = load_rules(rules_path)
    report = run_checks(batch_dir, rules)

    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    print(report_text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_text, encoding="utf-8")

    return 0 if report["summary"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
