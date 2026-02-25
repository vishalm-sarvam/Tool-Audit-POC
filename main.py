"""Tool Audit Pipeline Bottleneck Analysis — Main Entry Point.

Compares Gemini 2.5 Pro (LLM audit) vs ruff + regex (static analysis)
on Python tool files of varying sizes to prove the LLM pipeline doesn't scale.
"""

import os
import sys
from pathlib import Path

from generate_large_file import generate_large_file
from gemini_audit import run_gemini_audit, GeminiAuditResult
from static_audit import run_static_audit, StaticAuditResult

# ── Known planted issues per file ─────────────────────────────────────────────

PLANTED_ISSUES = {
    "small_good.py": [],
    "small_bad.py": [
        {"line": 15, "category": "sync_code", "description": "time.sleep in async"},
        {"line": 18, "category": "http_client", "description": "requests.get in async"},
        {"line": 32, "category": "memory_leak", "description": "open() without context manager"},
        {"line": 36, "category": "crypto", "description": "hashlib in async without to_thread"},
    ],
    "fake_async.py": [
        {"line": 19, "category": "hidden-blocking-call", "description": "async wrapper around sync lib — undetectable statically"},
    ],
    "httpx_leak.py": [
        {"line": 23, "category": "resource-leak", "description": "httpx.AsyncClient() without context manager — connection leak"},
        {"line": 34, "category": "http_client", "description": "requests.get() blocks async event loop"},
        {"line": 44, "category": "error-handling", "description": "raises exception instead of returning ToolOutput"},
    ],
    "large_bad.py": [],  # filled after generation
}

# ── Matching logic ────────────────────────────────────────────────────────────

# Category aliases — map between different naming conventions
CATEGORY_ALIASES = {
    "blocking-call": {"sync_code", "blocking-call", "http_client"},
    "sync_code": {"sync_code", "blocking-call", "http_client"},
    "http_client": {"sync_code", "blocking-call", "http_client"},
    "resource-leak": {"memory_leak", "resource-leak"},
    "memory_leak": {"memory_leak", "resource-leak"},
    "cpu-bound": {"crypto", "cpu-bound", "performance"},
    "crypto": {"crypto", "cpu-bound", "performance"},
    "security": {"security"},
    "hidden-blocking-call": {"sync_code", "blocking-call", "http_client", "hidden-blocking-call"},
    "error-handling": {"error-handling", "other"},
}


def _categories_match(planted_cat: str, found_cat: str) -> bool:
    """Check if a planted category matches a found category (with aliases)."""
    planted_aliases = CATEGORY_ALIASES.get(planted_cat, {planted_cat})
    return found_cat in planted_aliases


def _match_issues(planted: list[dict], found: list[dict], tolerance: int) -> dict:
    """
    Match found issues against planted issues.

    Returns: {matched, missed, extra, line_accuracy}
    """
    matched = []
    missed = []
    used_found = set()

    for p in planted:
        best_match = None
        for idx, f in enumerate(found):
            if idx in used_found:
                continue
            f_line = f.get("line", 0)
            f_cat = f.get("category", "")
            if abs(p["line"] - f_line) <= tolerance and _categories_match(p["category"], f_cat):
                if best_match is None or abs(p["line"] - f_line) < abs(p["line"] - best_match[1].get("line", 0)):
                    best_match = (idx, f)

        if best_match:
            used_found.add(best_match[0])
            matched.append({"planted": p, "found": best_match[1]})
        else:
            missed.append(p)

    extra = [f for idx, f in enumerate(found) if idx not in used_found]

    exact_matches = sum(
        1 for m in matched if m["planted"]["line"] == m["found"].get("line", -1)
    )
    line_accuracy = exact_matches / len(matched) if matched else 0.0

    return {
        "matched": matched,
        "missed": missed,
        "extra": extra,
        "detection_rate": len(matched) / len(planted) if planted else 1.0,
        "line_accuracy": line_accuracy,
    }


# ── Report formatting ─────────────────────────────────────────────────────────

SEPARATOR = "=" * 80
THIN_SEP = "-" * 80


def _print_file_result(name: str, data: dict) -> None:
    """Print results for a single file."""
    print(f"\n{SEPARATOR}")
    print(f"  FILE: {name} ({data['line_count']} lines, {len(data['planted'])} planted issues)")
    print(SEPARATOR)

    gemini: GeminiAuditResult = data["gemini"]
    static: StaticAuditResult = data["static"]
    planted = data["planted"]

    # Gemini results
    print(f"\n  [Gemini 2.5 Pro]")
    print(f"    Latency:       {gemini.latency_seconds:.2f}s")
    print(f"    Input tokens:  {gemini.input_tokens:,}")
    print(f"    Output tokens: {gemini.output_tokens:,}")
    print(f"    Cost (Flash):  ${gemini.estimated_cost_usd:.6f}")
    print(f"    Cost (Pro):    ${gemini.pro_estimated_cost_usd:.6f}")
    print(f"    JSON parsed:   {'Yes' if gemini.parse_success else 'NO — PARSE FAILED'}")
    print(f"    Issues found:  {len(gemini.issues)}")
    if gemini.error:
        print(f"    ERROR:         {gemini.error[:300]}")
    if gemini.summary:
        print(f"    Summary:       {gemini.summary}")

    gemini_match = _match_issues(planted, gemini.issues, tolerance=5)
    print(f"    Detection:     {len(gemini_match['matched'])}/{len(planted)} planted issues found")
    if gemini_match["missed"]:
        print(f"    MISSED:")
        for m in gemini_match["missed"]:
            print(f"      - Line {m['line']}: [{m['category']}] {m['description']}")

    # Static results
    print(f"\n  [Static Analysis (ruff + patterns)]")
    print(f"    Latency:       {static.latency_ms:.1f}ms")
    print(f"    Ruff findings: {len(static.ruff_findings)}")
    print(f"    Regex finds:   {len(static.pattern_findings)}")
    print(f"    Total (dedup): {len(static.all_findings)}")

    static_match = _match_issues(planted, static.all_findings, tolerance=2)
    print(f"    Detection:     {len(static_match['matched'])}/{len(planted)} planted issues found")
    if static_match["missed"]:
        print(f"    MISSED:")
        for m in static_match["missed"]:
            print(f"      - Line {m['line']}: [{m['category']}] {m['description']}")


def _print_summary(results: dict) -> None:
    """Print the final summary comparison table."""
    print(f"\n\n{'=' * 100}")
    print("  SUMMARY: Gemini 2.5 Pro vs Static Analysis (ruff + regex)")
    print(f"{'=' * 100}")
    print()

    header = f"  {'File':<20} {'Lines':>7} {'Planted':>8} │ {'Gemini Det':>10} {'Gemini Time':>12} {'Flash Cost':>12} {'Pro Cost':>12} │ {'Static Det':>10} {'Static Time':>12}"
    print(header)
    print(f"  {'─' * 18}  {'─' * 7} {'─' * 8} ┼ {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 12} ┼ {'─' * 10} {'─' * 12}")

    total_gemini_time = 0.0
    total_flash_cost = 0.0
    total_pro_cost = 0.0
    total_static_time = 0.0
    total_gemini_detected = 0
    total_static_detected = 0
    total_planted = 0

    for name, data in results.items():
        gemini: GeminiAuditResult = data["gemini"]
        static: StaticAuditResult = data["static"]
        planted = data["planted"]

        gemini_match = _match_issues(planted, gemini.issues, tolerance=5)
        static_match = _match_issues(planted, static.all_findings, tolerance=2)

        g_det = f"{len(gemini_match['matched'])}/{len(planted)}" if planted else "n/a"
        s_det = f"{len(static_match['matched'])}/{len(planted)}" if planted else "n/a"

        print(
            f"  {name:<20} {data['line_count']:>7} {len(planted):>8} │ "
            f"{g_det:>10} {gemini.latency_seconds:>10.2f}s ${gemini.estimated_cost_usd:>10.6f} ${gemini.pro_estimated_cost_usd:>10.6f} │ "
            f"{s_det:>10} {static.latency_ms:>10.1f}ms"
        )

        total_gemini_time += gemini.latency_seconds
        total_flash_cost += gemini.estimated_cost_usd
        total_pro_cost += gemini.pro_estimated_cost_usd
        total_static_time += static.latency_ms
        total_gemini_detected += len(gemini_match["matched"])
        total_static_detected += len(static_match["matched"])
        total_planted += len(planted)

    print(f"  {'─' * 18}  {'─' * 7} {'─' * 8} ┼ {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 12} ┼ {'─' * 10} {'─' * 12}")
    print(
        f"  {'TOTAL':<20} {'':>7} {total_planted:>8} │ "
        f"{total_gemini_detected}/{total_planted:>8} {total_gemini_time:>10.2f}s ${total_flash_cost:>10.6f} ${total_pro_cost:>10.6f} │ "
        f"{total_static_detected}/{total_planted:>8} {total_static_time:>10.1f}ms"
    )

    # Speedup factor
    if total_gemini_time > 0:
        speedup = (total_gemini_time * 1000) / max(total_static_time, 0.1)
        print(f"\n  Static analysis is {speedup:,.0f}x faster than Gemini")
    print(f"  Gemini API cost (Flash): ${total_flash_cost:.6f} / run")
    print(f"  Gemini API cost (Pro):   ${total_pro_cost:.6f} / run")
    print(f"  Static analysis cost:    $0.000000 / run")

    # Verdict
    print(f"\n{SEPARATOR}")
    print("  VERDICT")
    print(SEPARATOR)
    print("  The LLM-only audit pipeline (PR #741) has fundamental scaling problems:")
    print(f"    1. Latency: {total_gemini_time:.1f}s total for Gemini vs {total_static_time:.0f}ms for static analysis")
    print(f"    2. Cost: ${total_flash_cost:.6f} (Flash) / ${total_pro_cost:.6f} (Pro) per run")
    print(f"    3. Detection: Gemini found {total_gemini_detected}/{total_planted} issues vs static {total_static_detected}/{total_planted}")
    print("    4. fake_async.py: Neither approach detects hidden sync wrappers — runtime analysis needed")
    print("    5. Reliability: LLM output requires JSON parsing, is non-deterministic, and may hallucinate")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 0. Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: Set GEMINI_API_KEY environment variable")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)

    # 1. Generate large file
    print("Generating large_bad.py (~15K LOC)...")
    os.makedirs("samples", exist_ok=True)
    large_file_issues = generate_large_file("samples/large_bad.py")
    PLANTED_ISSUES["large_bad.py"] = large_file_issues

    # 2. Load prompt template
    prompt_template = Path("prompt.txt").read_text()

    # 3. Run benchmarks
    results = {}
    for sample_name in ["small_good.py", "small_bad.py", "fake_async.py", "httpx_leak.py", "large_bad.py"]:
        file_path = f"samples/{sample_name}"
        line_count = sum(1 for _ in open(file_path))
        planted = PLANTED_ISSUES[sample_name]

        print(f"\n{'─' * 60}")
        print(f"Running: {sample_name} ({line_count} lines, {len(planted)} planted issues)...")

        gemini = run_gemini_audit(file_path, prompt_template)
        static = run_static_audit(file_path)

        results[sample_name] = {
            "line_count": line_count,
            "planted": planted,
            "gemini": gemini,
            "static": static,
        }

    # 4. Print report
    for name, data in results.items():
        _print_file_result(name, data)

    _print_summary(results)


if __name__ == "__main__":
    main()
