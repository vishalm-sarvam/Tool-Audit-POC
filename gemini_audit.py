"""Gemini 2.5 Pro audit module — sends tool code for LLM-based review."""

import json
import os
import re
import time
from dataclasses import dataclass, field

from google import genai
from google.genai.types import GenerateContentConfig


@dataclass
class GeminiAuditResult:
    issues: list[dict] = field(default_factory=list)
    raw_response: str = ""
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    pro_estimated_cost_usd: float = 0.0
    parse_success: bool = False
    status: str = "UNKNOWN"
    summary: str = ""
    error: str = ""


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences robustly."""
    cleaned = text.strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences — handle ```json\n...\n``` and ```\n...\n```
    # Use greedy match on the inner content to get the largest JSON block
    match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block (outermost braces)
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {cleaned[:300]}...")


def run_gemini_audit(file_path: str, prompt_template: str) -> GeminiAuditResult:
    """
    Send file to Gemini 2.5 Pro for audit.

    Uses Gemini 2.5 Pro with its 1M token context window to handle large files.
    """
    result = GeminiAuditResult()

    # Read the tool code
    with open(file_path) as f:
        tool_code = f.read()

    # Build the prompt — template uses {{}} for literal braces, {{tool_code}} for substitution
    prompt = prompt_template.replace("{{tool_code}}", tool_code)

    # Configure API
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        result.raw_response = "ERROR: GEMINI_API_KEY not set"
        result.error = "GEMINI_API_KEY not set"
        return result

    client = genai.Client(api_key=api_key)

    # Try models in order: Pro first (1M context), Flash as fallback
    models_to_try = ["gemini-2.5-pro", "gemini-2.5-flash"]
    response = None
    used_model = None

    start = time.perf_counter()
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=8192,
                ),
            )
            used_model = model_name
            break
        except Exception as e:
            error_str = str(e)
            # If rate limited or quota exceeded, try next model
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(f"    {model_name}: rate limited, trying fallback...")
                time.sleep(1)
                continue
            # For other errors, record and return
            result.latency_seconds = time.perf_counter() - start
            result.raw_response = f"API ERROR ({model_name}): {e}"
            result.error = error_str
            return result

    result.latency_seconds = time.perf_counter() - start

    if response is None:
        result.error = "All models exhausted (rate limited on both Pro and Flash)"
        result.raw_response = "ERROR: All models rate limited"
        return result

    result.summary = f"[model={used_model}] "

    # Check for empty/blocked response
    if not response.candidates:
        result.error = "No candidates in response (possibly blocked or empty)"
        result.raw_response = str(response)
        return result

    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        result.error = f"Empty response content. Finish reason: {candidate.finish_reason}"
        result.raw_response = str(response)
        return result

    # Extract token counts from usage_metadata
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        result.input_tokens = getattr(meta, "prompt_token_count", 0) or 0
        result.output_tokens = getattr(meta, "candidates_token_count", 0) or 0

    # Calculate cost based on which model was used
    # Pricing source: https://ai.google.dev/gemini-api/docs/pricing
    if used_model == "gemini-2.5-pro":
        # Pro pricing: Input $1.25/1M (<200K), $2.50/1M (>200K), Output $10.00/1M
        if result.input_tokens <= 200_000:
            input_cost = result.input_tokens * 1.25 / 1_000_000
        else:
            input_cost = (
                200_000 * 1.25 / 1_000_000
                + (result.input_tokens - 200_000) * 2.50 / 1_000_000
            )
        output_cost = result.output_tokens * 10.00 / 1_000_000
    else:
        # Flash pricing: Input $0.30/1M, Output $2.50/1M
        input_cost = result.input_tokens * 0.30 / 1_000_000
        output_cost = result.output_tokens * 2.50 / 1_000_000
    result.estimated_cost_usd = input_cost + output_cost

    # Also calculate what Pro would have cost (for comparison in report)
    if result.input_tokens <= 200_000:
        pro_input_cost = result.input_tokens * 1.25 / 1_000_000
    else:
        pro_input_cost = (
            200_000 * 1.25 / 1_000_000
            + (result.input_tokens - 200_000) * 2.50 / 1_000_000
        )
    pro_output_cost = result.output_tokens * 10.00 / 1_000_000
    result.pro_estimated_cost_usd = pro_input_cost + pro_output_cost

    # Extract response text
    try:
        result.raw_response = response.text
    except Exception:
        result.raw_response = str(candidate.content.parts[0].text if candidate.content.parts else "")

    if not result.raw_response.strip():
        result.error = "Empty response text"
        return result

    # Parse JSON
    model_prefix = f"[model={used_model}] "
    try:
        parsed = _extract_json(result.raw_response)
        result.issues = parsed.get("issues", [])
        result.status = parsed.get("status", "UNKNOWN")
        result.summary = model_prefix + parsed.get("summary", "")
        result.parse_success = True
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        result.parse_success = False
        result.error = f"JSON parse failed: {e}"
        result.summary = model_prefix + f"JSON parse failed: {e}"

    return result
