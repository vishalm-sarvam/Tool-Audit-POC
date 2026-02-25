# Tool Audit Pipeline Bottleneck Analysis — Full Context

## Purpose
Prove that PR #741's LLM-only tool audit pipeline doesn't scale. Compare Gemini 2.5 Flash vs ruff static analysis on Python tool files of varying sizes.

## Project Structure
```
tool-audit-poc/
├── main.py                  # Entry point — runs all benchmarks, prints report
├── gemini_audit.py          # Gemini API caller with the audit prompt
├── static_audit.py          # ruff-based static analysis runner
├── generate_large_file.py   # Generates a realistic 15K LOC tool file
├── prompt.txt               # The audit prompt template
├── samples/
│   ├── small_good.py        # ~50 LOC, clean async code
│   ├── small_bad.py         # ~50 LOC, intentional issues
│   ├── fake_async.py        # ~30 LOC, looks async but uses sync lib internally
│   └── (large_bad.py)       # Generated at runtime
├── ruff.toml                # ruff config
├── requirements.txt
└── README.md
```

---

## File Contents

---

### requirements.txt
```
google-generativeai>=0.8.0
ruff>=0.8.0
```

---

### ruff.toml
```toml
[lint]
select = ["ASYNC", "S", "BLE", "E", "W", "F"]
```

---

### prompt.txt
This is the audit prompt sent to Gemini. `{tool_code}` is replaced at runtime with the file contents.

```
You are a senior Python code reviewer specializing in async safety and security for LangChain-based tools.

Analyze the following Python tool code and return a JSON response with this exact structure:
{{
  "issues": [
    {{
      "line": <line_number>,
      "severity": "error" | "warning" | "info",
      "category": "blocking-call" | "security" | "resource-leak" | "cpu-bound" | "best-practice",
      "message": "<description of the issue>"
    }}
  ],
  "summary": "<one paragraph overall assessment>",
  "safe_to_deploy": true | false
}}

Rules:
1. Flag any blocking/synchronous calls inside async functions (time.sleep, requests.*, subprocess.run without async, etc.)
2. Flag file operations without context managers (open() without with statement)
3. Flag CPU-bound operations in async context without asyncio.to_thread (hashlib, heavy computation)
4. Flag security issues (eval, exec, os.system, shell=True, hardcoded secrets)
5. Flag broad exception handling (bare except:, except Exception:)
6. Return ONLY valid JSON — no markdown, no explanation outside the JSON.

Tool code to review:
```python
{tool_code}
```
```

---

### samples/small_good.py
```python
"""A well-written async tool — no issues expected."""

import asyncio
import hashlib
from typing import Optional

import httpx
import aiofiles
from langchain.tools import BaseTool


class AsyncWebFetcher(BaseTool):
    name: str = "async_web_fetcher"
    description: str = "Fetches a URL asynchronously and returns the content hash."

    async def _arun(self, url: str, save_path: Optional[str] = None) -> str:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.text

        content_hash = await asyncio.to_thread(hashlib.sha256, content.encode())

        if save_path:
            async with aiofiles.open(save_path, "w") as f:
                await f.write(content)

        return f"Fetched {len(content)} chars, hash={content_hash.hexdigest()}"

    def _run(self, url: str, save_path: Optional[str] = None) -> str:
        raise NotImplementedError("Use async version")


class AsyncDataProcessor(BaseTool):
    name: str = "async_data_processor"
    description: str = "Processes data entries asynchronously."

    async def _arun(self, entries: list[str]) -> dict:
        results = {}
        for entry in entries:
            await asyncio.sleep(0)  # yield control
            results[entry] = len(entry)
        return {"processed": len(results), "results": results}

    def _run(self, entries: list[str]) -> dict:
        raise NotImplementedError("Use async version")
```

---

### samples/small_bad.py
```python
"""A poorly-written async tool — multiple intentional issues planted."""

import time
import hashlib
import requests
from langchain.tools import BaseTool


class BadWebFetcher(BaseTool):
    name: str = "bad_web_fetcher"
    description: str = "Fetches a URL (badly)."

    async def _arun(self, url: str) -> str:
        # ISSUE 1 (line 15): blocking sleep in async function
        time.sleep(1)

        # ISSUE 2 (line 18): sync HTTP call in async function
        response = requests.get(url, timeout=30)
        data = response.text

        return f"Fetched {len(data)} chars"

    def _run(self, url: str) -> str:
        raise NotImplementedError("Use async version")


class BadFileProcessor(BaseTool):
    name: str = "bad_file_processor"
    description: str = "Processes a file (badly)."

    async def _arun(self, file_path: str) -> str:
        # ISSUE 3 (line 32): open() without context manager
        f = open(file_path)
        content = f.read()
        f.close()

        # ISSUE 4 (line 36): CPU-bound crypto in async without to_thread
        digest = hashlib.sha256(content.encode()).hexdigest()

        return f"Hash: {digest}"

    def _run(self, file_path: str) -> str:
        raise NotImplementedError("Use async version")
```

**Planted issues registry for small_bad.py:**
```python
SMALL_BAD_ISSUES = [
    {"line": 15, "category": "blocking-call", "description": "time.sleep in async"},
    {"line": 18, "category": "blocking-call", "description": "requests.get in async"},
    {"line": 32, "category": "resource-leak", "description": "open() without context manager"},
    {"line": 36, "category": "cpu-bound", "description": "hashlib in async without to_thread"},
]
```

---

### samples/fake_async.py
```python
"""THE KEY TEST CASE: Looks async but uses a sync library internally.
Neither Gemini nor ruff can catch this without runtime analysis.
This disproves the argument that 'Gemini catches sync 3rd-party libs'."""

from langchain.tools import BaseTool


class AsyncFetcher:
    """Pretend third-party library. Looks async, but internally uses requests.get()."""
    async def get(self, url: str) -> str:
        # In reality, this would internally call requests.get(url)
        # But from the caller's perspective, it's an awaitable
        return f"response from {url}"


class FakeAsyncTool(BaseTool):
    name: str = "fake_async_tool"
    description: str = "Uses a third-party async wrapper that internally blocks."

    async def _arun(self, url: str) -> str:
        fetcher = AsyncFetcher()
        # HIDDEN ISSUE (line 22): fetcher.get() LOOKS async but internally does requests.get()
        # Neither Gemini nor ruff can detect this — it requires runtime analysis or
        # inspecting the third-party library's source code.
        result = await fetcher.get(url)
        return result

    def _run(self, url: str) -> str:
        raise NotImplementedError("Use async version")
```

**Planted issues registry for fake_async.py:**
```python
FAKE_ASYNC_ISSUES = [
    {"line": 22, "category": "hidden-blocking-call", "description": "async wrapper around sync lib — undetectable by static analysis or LLM"},
]
```

---

### generate_large_file.py

**Goal:** Generate `samples/large_bad.py` with ~15,000 lines. Bulk is legitimate-looking async tool classes (padding). 5 issues planted at known line numbers.

**Strategy:**
- Generate ~30 tool classes, each ~480 lines of realistic async code (helper methods, docstrings, type hints, error handling)
- Plant 5 specific issues at approximately lines 200, 3000, 7500, 11000, 14500
- Return exact planted line numbers for verification

**Planted issues for large_bad.py:**
```python
LARGE_BAD_ISSUES = [
    {"line": "~200", "category": "blocking-call", "description": "time.sleep(2) in async method"},
    {"line": "~3000", "category": "blocking-call", "description": "requests.post() in async method"},
    {"line": "~7500", "category": "resource-leak", "description": "open() without context manager"},
    {"line": "~11000", "category": "security", "description": "eval() on user input"},
    {"line": "~14500", "category": "cpu-bound", "description": "hashlib.pbkdf2_hmac in async without to_thread"},
]
```

**Padding class template** (each class ~480 lines):
```python
class Tool_{n}(BaseTool):
    name: str = "tool_{n}"
    description: str = "Async tool variant {n} for data processing."

    async def _arun(self, input_data: str) -> dict:
        """Process input data asynchronously."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://api.example.com/v1/data/{n}")
            response.raise_for_status()
        # ... lots of legitimate async processing, string manipulation,
        # dict building, list comprehensions, logging calls, etc.
        results = {}
        for i in range({n*10}, {n*10+50}):
            await asyncio.sleep(0)
            results[f"key_{i}"] = f"value_{i}_{input_data[:10]}"
        return {"tool": "tool_{n}", "results": results}

    def _run(self, input_data: str) -> dict:
        raise NotImplementedError("Use async version")
```

Each padding class should include:
- Multiple helper methods (async and sync)
- Docstrings, type annotations
- Realistic async patterns (httpx, aiofiles, asyncio.gather)
- Data processing logic (dicts, lists, string ops)

---

### gemini_audit.py

**Interface:**
```python
import time
import json
import os
import google.generativeai as genai
from dataclasses import dataclass

@dataclass
class GeminiAuditResult:
    issues: list[dict]          # parsed issues from JSON response
    raw_response: str           # raw LLM output
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    parse_success: bool         # whether JSON parsing succeeded

def run_gemini_audit(file_path: str, prompt_template: str) -> GeminiAuditResult:
    """
    Send file to Gemini 2.5 Flash for audit.

    - Read prompt.txt, replace {tool_code} with file contents
    - Call gemini-2.5-flash via google.generativeai SDK
    - Measure wall-clock latency
    - Extract token counts from response metadata (usage_metadata)
    - Calculate cost: input=$0.15/1M tokens, output=$0.60/1M tokens (Flash pricing)
    - Parse JSON from response (handle markdown ```json ... ``` wrapping)
    - Return GeminiAuditResult
    """
```

**Key implementation details:**
- Model: `gemini-2.5-flash`
- API key from `os.environ["GEMINI_API_KEY"]`
- JSON extraction: strip markdown fences if present (`response.text.strip()`, then regex for ```json...```)
- Token counts from `response.usage_metadata.prompt_token_count` and `.candidates_token_count`
- Cost calculation: Gemini 2.5 Flash pricing = $0.15/1M input, $0.60/1M output (adjust if needed)

---

### static_audit.py

**Interface:**
```python
import subprocess
import json
import re
import time
from dataclasses import dataclass

@dataclass
class StaticAuditResult:
    ruff_findings: list[dict]    # from ruff JSON output
    pattern_findings: list[dict] # from regex pattern matching
    all_findings: list[dict]     # combined, deduplicated
    latency_ms: float

def run_static_audit(file_path: str, ruff_config: str = "ruff.toml") -> StaticAuditResult:
    """
    Run ruff + custom regex patterns on file.

    1. Run: ruff check --config ruff.toml --output-format json <file_path>
       Parse JSON output for findings.

    2. Custom regex patterns (catches things ruff may miss):
       - r'time\.sleep\(' → blocking sleep
       - r'requests\.(get|post|put|delete|patch|head)\(' → sync HTTP
       - r'^(\s*)(\w+)\s*=\s*open\(' (without preceding 'with') → resource leak
       - r'hashlib\.\w+\(' in async function context → CPU-bound
       - r'\beval\(' → security
       - r'\bexec\(' → security
       - r'\bos\.system\(' → security

    3. Combine and deduplicate (same line + similar category = one finding)
    4. Measure total wall-clock time in ms
    """
```

**ruff command:**
```bash
ruff check --config ruff.toml --output-format json <file_path>
```

Parse the JSON array. Each entry has `location.row`, `code`, `message`.

---

### main.py

**Flow:**
```python
import os
import sys
from pathlib import Path

from generate_large_file import generate_large_file
from gemini_audit import run_gemini_audit
from static_audit import run_static_audit

# Known planted issues per file
PLANTED_ISSUES = {
    "small_bad.py": [...],   # 4 issues with exact lines
    "small_good.py": [],     # 0 issues
    "fake_async.py": [...],  # 1 hidden issue
    "large_bad.py": [...],   # 5 issues (lines filled after generation)
}

def compare_results(planted, gemini_result, static_result):
    """
    For each planted issue, check if Gemini/ruff found it.
    Match criteria: same line number (±5 lines tolerance for Gemini, exact for ruff)
    and same category.
    Returns detection counts and line accuracy.
    """

def print_report(all_results):
    """Print the formatted comparison report as shown in the plan's Expected Output."""

def main():
    # 0. Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    # 1. Generate large file
    large_file_issues = generate_large_file("samples/large_bad.py")
    PLANTED_ISSUES["large_bad.py"] = large_file_issues

    # 2. Load prompt template
    prompt_template = Path("prompt.txt").read_text()

    # 3. Run benchmarks for each sample
    results = {}
    for sample_name in ["small_good.py", "small_bad.py", "fake_async.py", "large_bad.py"]:
        file_path = f"samples/{sample_name}"
        line_count = sum(1 for _ in open(file_path))
        planted = PLANTED_ISSUES[sample_name]

        print(f"\nRunning: {sample_name} ({line_count} lines, {len(planted)} planted issues)...")

        gemini = run_gemini_audit(file_path, prompt_template)
        static = run_static_audit(file_path)

        results[sample_name] = {
            "line_count": line_count,
            "planted": planted,
            "gemini": gemini,
            "static": static,
        }

    # 4. Print report
    print_report(results)

if __name__ == "__main__":
    main()
```

**Report format:** Follow the expected output format from the plan — test-by-test breakdown, then summary table with metrics comparison.

**Matching logic for comparing found vs planted issues:**
- For Gemini: match if found issue is within ±5 lines of planted issue AND same category
- For ruff/patterns: match if found issue is within ±2 lines of planted issue AND same category
- Track: issues_found / issues_planted, and line_accuracy (exact matches / total found)

---

## Key Design Decisions

1. **Gemini model:** `gemini-2.5-flash` — same as what the PR would use in production
2. **Cost calculation:** Based on published Gemini 2.5 Flash pricing
3. **Tolerance for line matching:** Gemini gets ±5 lines tolerance (LLMs are imprecise), ruff gets ±2
4. **The fake_async test is the critical argument killer** — it proves that Gemini's "advantage" of understanding async semantics doesn't extend to third-party library internals
5. **Large file is the scaling proof** — 15K LOC will show massive latency AND degraded detection quality

## Environment
- Python 3.11+
- API key via `GEMINI_API_KEY` env var
- Run from project root: `cd ~/Projects/tool-audit-poc && python main.py`
