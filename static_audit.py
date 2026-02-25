"""Static analysis module — ruff + custom regex pattern matching."""

import json
import re
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class StaticAuditResult:
    ruff_findings: list[dict] = field(default_factory=list)
    pattern_findings: list[dict] = field(default_factory=list)
    all_findings: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0


# Regex patterns to catch common async/security issues that ruff may miss
PATTERNS = [
    {
        "regex": r"time\.sleep\(",
        "category": "sync_code",
        "severity": "critical",
        "message": "Blocking time.sleep() call — use asyncio.sleep() in async context",
    },
    {
        "regex": r"requests\.(get|post|put|delete|patch|head)\(",
        "category": "http_client",
        "severity": "critical",
        "message": "Synchronous requests library call — use httpx.AsyncClient in async context",
    },
    {
        "regex": r"(?<!with\s)open\(",
        "category": "memory_leak",
        "severity": "high",
        "message": "open() possibly without context manager — risk of resource leak",
    },
    {
        "regex": r"hashlib\.\w+\(",
        "category": "crypto",
        "severity": "medium",
        "message": "Synchronous hashlib call — use asyncio.to_thread() in async context",
    },
    {
        "regex": r"\beval\(",
        "category": "security",
        "severity": "critical",
        "message": "eval() call — potential code injection vulnerability",
    },
    {
        "regex": r"\bexec\(",
        "category": "security",
        "severity": "critical",
        "message": "exec() call — potential code injection vulnerability",
    },
    {
        "regex": r"\bos\.system\(",
        "category": "security",
        "severity": "critical",
        "message": "os.system() call — use subprocess with shell=False instead",
    },
    {
        "regex": r"subprocess\.\w+\(.*shell\s*=\s*True",
        "category": "security",
        "severity": "high",
        "message": "subprocess with shell=True — potential command injection",
    },
    {
        "regex": r"^\s*client\s*=\s*httpx\.AsyncClient\(",
        "category": "resource-leak",
        "severity": "high",
        "message": "httpx.AsyncClient() without context manager — connection pool leak (use 'async with')",
    },
    {
        "regex": r"raise\s+(ValueError|TypeError|RuntimeError|Exception)\(",
        "category": "error-handling",
        "severity": "medium",
        "message": "Raising exception in tool — should return ToolOutput with error message instead",
    },
]


def _run_ruff(file_path: str, ruff_config: str = "ruff.toml") -> list[dict]:
    """Run ruff check and parse JSON output."""
    try:
        proc = subprocess.run(
            [
                "ruff", "check",
                "--config", ruff_config,
                "--output-format", "json",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        print("  WARNING: ruff not found in PATH, skipping ruff analysis")
        return []
    except subprocess.TimeoutExpired:
        print("  WARNING: ruff timed out after 60s")
        return []

    # ruff exits with 1 when it finds issues, that's expected
    output = proc.stdout.strip()
    if not output:
        return []

    try:
        raw_findings = json.loads(output)
    except json.JSONDecodeError:
        print(f"  WARNING: Could not parse ruff output: {output[:200]}")
        return []

    findings = []
    for item in raw_findings:
        findings.append({
            "line": item.get("location", {}).get("row", 0),
            "code": item.get("code", ""),
            "category": _ruff_code_to_category(item.get("code", "")),
            "severity": _ruff_code_to_severity(item.get("code", "")),
            "message": item.get("message", ""),
            "source": "ruff",
        })
    return findings


def _ruff_code_to_category(code: str) -> str:
    """Map ruff rule codes to our category taxonomy."""
    if code.startswith("ASYNC"):
        return "sync_code"
    if code.startswith("S"):
        return "security"
    if code.startswith("BLE"):
        return "other"
    return "other"


def _ruff_code_to_severity(code: str) -> str:
    """Map ruff rule codes to severity levels."""
    if code.startswith("S"):
        return "high"
    if code.startswith("ASYNC"):
        return "critical"
    return "medium"


def _is_comment_or_docstring(line: str) -> bool:
    """Check if a line is a comment or inside a docstring marker."""
    stripped = line.lstrip()
    return stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''")


def _run_patterns(file_path: str) -> list[dict]:
    """Run custom regex patterns against file contents."""
    with open(file_path) as f:
        lines = f.readlines()

    findings = []
    for line_num, line_text in enumerate(lines, start=1):
        # Skip comments and docstrings to avoid false positives
        if _is_comment_or_docstring(line_text):
            continue
        for pattern in PATTERNS:
            if re.search(pattern["regex"], line_text):
                # Skip aiofiles.open — it's the correct async pattern
                if pattern["category"] == "memory_leak" and "aiofiles.open" in line_text:
                    continue
                findings.append({
                    "line": line_num,
                    "category": pattern["category"],
                    "severity": pattern["severity"],
                    "message": pattern["message"],
                    "source": "pattern",
                    "matched_text": line_text.strip(),
                })
    return findings


def _deduplicate(ruff_findings: list[dict], pattern_findings: list[dict]) -> list[dict]:
    """Combine and deduplicate findings (same line + overlapping category = one entry)."""
    combined = []
    seen = set()

    # Prefer ruff findings (more precise)
    for f in ruff_findings:
        key = (f["line"], f["category"])
        if key not in seen:
            seen.add(key)
            combined.append(f)

    # Add pattern findings that don't overlap
    for f in pattern_findings:
        key = (f["line"], f["category"])
        if key not in seen:
            seen.add(key)
            combined.append(f)

    return sorted(combined, key=lambda x: x["line"])


def run_static_audit(file_path: str, ruff_config: str = "ruff.toml") -> StaticAuditResult:
    """
    Run ruff + custom regex patterns on a file.

    Returns combined, deduplicated findings with timing.
    """
    start = time.perf_counter()

    ruff_findings = _run_ruff(file_path, ruff_config)
    pattern_findings = _run_patterns(file_path)
    all_findings = _deduplicate(ruff_findings, pattern_findings)

    latency_ms = (time.perf_counter() - start) * 1000

    return StaticAuditResult(
        ruff_findings=ruff_findings,
        pattern_findings=pattern_findings,
        all_findings=all_findings,
        latency_ms=latency_ms,
    )
