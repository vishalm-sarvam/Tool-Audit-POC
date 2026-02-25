"""Generate samples/large_bad.py — a ~15K LOC file with 5 planted issues buried in padding."""

import textwrap


def _padding_class(n: int) -> str:
    """Generate a single realistic-looking async tool class (~580 lines)."""
    base_range = n * 10
    return textwrap.dedent(f"""\

        class Tool_{n}(BaseTool):
            \"\"\"Async tool variant {n} for data processing.

            This tool handles batch data processing operations with configurable
            parallelism and retry logic. It connects to external APIs and processes
            results through a multi-stage pipeline.

            Attributes:
                name: Unique tool identifier used for routing.
                description: Human-readable description shown in tool listings.
                max_retries: Maximum number of retry attempts for failed operations.
                batch_size: Number of items to process in each batch.
                timeout_seconds: Default timeout for HTTP operations.
            \"\"\"
            name: str = "tool_{n}"
            description: str = "Async tool variant {n} for data processing."
            max_retries: int = 3
            batch_size: int = 50
            timeout_seconds: int = 30

            async def _arun(self, input_data: str) -> dict:
                \"\"\"Process input data asynchronously.

                This method orchestrates the full processing pipeline:
                1. Fetch reference data from the external API
                2. Build initial result set from input
                3. Transform and validate results
                4. Compute statistics and return

                Args:
                    input_data: Raw input string to process.

                Returns:
                    Dictionary with processing results, metadata, and statistics.

                Raises:
                    httpx.HTTPStatusError: If the API call fails.
                    ValueError: If input_data is empty.
                \"\"\"
                if not input_data:
                    raise ValueError("input_data cannot be empty")

                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.get(f"https://api.example.com/v1/data/{n}")
                    response.raise_for_status()
                    api_data = response.json()

                results = {{}}
                for i in range({base_range}, {base_range + 50}):
                    await asyncio.sleep(0)
                    results[f"key_{{i}}"] = f"value_{{i}}_{{input_data[:10]}}"

                processed = await self._transform_results(results, api_data)
                validated = await self._validate_output(processed)
                stats = await self._compute_stats(validated)
                return {{
                    "tool": "tool_{n}",
                    "results": validated,
                    "count": len(results),
                    "stats": stats,
                }}

            async def _transform_results(self, results: dict, api_data: dict) -> dict:
                \"\"\"Apply transformations to raw results.

                Each result entry is enriched with length metadata and an API
                reference identifier for traceability.

                Args:
                    results: Raw key-value results from initial processing.
                    api_data: Reference data from the external API.

                Returns:
                    Transformed dictionary with enriched entries.
                \"\"\"
                transformed = {{}}
                for key, value in results.items():
                    await asyncio.sleep(0)
                    transformed[key] = {{
                        "original": value,
                        "length": len(value),
                        "api_ref": api_data.get("ref", "unknown"),
                        "processed": True,
                    }}
                return transformed

            async def _validate_output(self, data: dict) -> dict:
                \"\"\"Validate processed output against expected schema.

                Filters out entries that don't conform to the expected structure.
                Valid entries must be dictionaries containing an 'original' key.

                Args:
                    data: Transformed data to validate.

                Returns:
                    Dictionary containing only valid entries.
                \"\"\"
                validated = {{}}
                for key, value in data.items():
                    if isinstance(value, dict) and "original" in value:
                        validated[key] = value
                    else:
                        pass  # silently drop invalid entries
                return validated

            async def _fetch_metadata(self, item_id: str) -> dict:
                \"\"\"Fetch metadata for a specific item from the metadata service.

                Args:
                    item_id: Unique identifier for the item.

                Returns:
                    Metadata dictionary from the service.
                \"\"\"
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(
                        f"https://api.example.com/v1/metadata/{{item_id}}"
                    )
                    resp.raise_for_status()
                    return resp.json()

            async def _batch_process(self, items: list[str]) -> list[dict]:
                \"\"\"Process a batch of items concurrently using asyncio.gather.

                Args:
                    items: List of item IDs to process.

                Returns:
                    List of metadata dictionaries, one per item.
                \"\"\"
                tasks = [self._fetch_metadata(item) for item in items]
                return await asyncio.gather(*tasks)

            async def _retry_operation(self, coro, max_retries: int = 3):
                \"\"\"Retry an async operation with exponential backoff.

                Args:
                    coro: The coroutine to retry.
                    max_retries: Maximum number of attempts.

                Returns:
                    The result of the coroutine if successful.

                Raises:
                    httpx.HTTPStatusError: If all retries are exhausted.
                \"\"\"
                for attempt in range(max_retries):
                    try:
                        return await coro
                    except httpx.HTTPStatusError:
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(0.1 * (2 ** attempt))

            async def _stream_results(self, query: str) -> list[str]:
                \"\"\"Stream results from API endpoint line by line.

                Args:
                    query: Search query to stream results for.

                Returns:
                    List of result lines from the streaming response.
                \"\"\"
                results = []
                async with httpx.AsyncClient(timeout=60) as client:
                    async with client.stream(
                        "GET", f"https://api.example.com/v1/stream/{{query}}"
                    ) as response:
                        async for line in response.aiter_lines():
                            results.append(line)
                return results

            async def _compute_stats(self, data: dict) -> dict:
                \"\"\"Compute descriptive statistics for processed data.

                Calculates count, average, min, and max of string representations
                of values in the data dictionary.

                Args:
                    data: Dictionary of processed data entries.

                Returns:
                    Statistics dictionary with total, avg, min, max lengths.
                \"\"\"
                total = len(data)
                if total == 0:
                    return {{"total_items": 0, "avg_length": 0, "min_length": 0, "max_length": 0}}
                lengths = [len(str(v)) for v in data.values()]
                avg_len = sum(lengths) / total
                return {{
                    "total_items": total,
                    "avg_length": round(avg_len, 2),
                    "min_length": min(lengths),
                    "max_length": max(lengths),
                    "total_length": sum(lengths),
                }}

            async def _filter_results(self, data: dict, threshold: int = 10) -> dict:
                \"\"\"Filter results based on string length threshold.

                Args:
                    data: Dictionary of results to filter.
                    threshold: Minimum string length to include.

                Returns:
                    Filtered dictionary containing only entries above threshold.
                \"\"\"
                filtered = {{}}
                for key, value in data.items():
                    await asyncio.sleep(0)
                    if len(str(value)) > threshold:
                        filtered[key] = value
                return filtered

            async def _merge_datasets(self, primary: dict, secondary: dict) -> dict:
                \"\"\"Merge two datasets with conflict resolution.

                When keys conflict, the secondary entry is stored with a '_secondary' suffix.

                Args:
                    primary: Primary dataset (takes precedence).
                    secondary: Secondary dataset.

                Returns:
                    Merged dictionary with conflict resolution applied.
                \"\"\"
                merged = dict(primary)
                for key, value in secondary.items():
                    if key not in merged:
                        merged[key] = value
                    else:
                        merged[f"{{key}}_secondary"] = value
                return merged

            async def _format_output(self, data: dict) -> str:
                \"\"\"Format output data as a human-readable structured string.

                Args:
                    data: Dictionary to format.

                Returns:
                    Formatted string with sorted key-value pairs.
                \"\"\"
                lines = []
                for key, value in sorted(data.items()):
                    lines.append(f"  {{key}}: {{value}}")
                header = f"Tool_{n} Output ({{len(data)}} entries):"
                return header + "\\n" + "\\n".join(lines)

            async def _log_operation(self, operation: str, details: dict) -> None:
                \"\"\"Log an operation with details to an async-safe log file.

                Args:
                    operation: Name of the operation being logged.
                    details: Dictionary of operation details.
                \"\"\"
                log_entry = {{
                    "operation": operation,
                    "details": details,
                    "tool": self.name,
                }}
                async with aiofiles.open(f"/tmp/tool_{n}.log", "a") as f:
                    await f.write(str(log_entry) + "\\n")

            async def _cleanup(self, resources: list) -> None:
                \"\"\"Clean up resources after processing.

                Iterates through resources and calls aclose() on any that support it.

                Args:
                    resources: List of resources to clean up.
                \"\"\"
                for resource in resources:
                    if hasattr(resource, "aclose"):
                        await resource.aclose()

            async def _paginate_results(
                self, endpoint: str, page_size: int = 50
            ) -> list[dict]:
                \"\"\"Paginate through API results fetching all pages.

                Args:
                    endpoint: API endpoint URL to paginate.
                    page_size: Number of items per page.

                Returns:
                    Concatenated list of all items across all pages.
                \"\"\"
                all_results = []
                page = 1
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    while True:
                        resp = await client.get(
                            endpoint, params={{"page": page, "size": page_size}}
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        if not data.get("items"):
                            break
                        all_results.extend(data["items"])
                        page += 1
                return all_results

            async def _deduplicate(self, items: list[dict]) -> list[dict]:
                \"\"\"Remove duplicate items based on their ID field.

                Args:
                    items: List of dictionaries to deduplicate.

                Returns:
                    List of unique items preserving first occurrence order.
                \"\"\"
                seen = set()
                unique = []
                for item in items:
                    item_id = item.get("id", str(item))
                    if item_id not in seen:
                        seen.add(item_id)
                        unique.append(item)
                return unique

            async def _enrich_data(self, items: list[dict]) -> list[dict]:
                \"\"\"Enrich items with additional metadata from the metadata service.

                Args:
                    items: List of items to enrich.

                Returns:
                    List of items with metadata added.
                \"\"\"
                enriched = []
                for item in items:
                    metadata = await self._fetch_metadata(item.get("id", "unknown"))
                    enriched.append({{**item, "metadata": metadata}})
                return enriched

            async def _cache_results(self, key: str, data: dict) -> None:
                \"\"\"Cache results to a local temporary file.

                Args:
                    key: Cache key used for the filename.
                    data: Data to cache.
                \"\"\"
                async with aiofiles.open(f"/tmp/cache_{{key}}.json", "w") as f:
                    await f.write(str(data))

            async def _normalize_keys(self, data: dict) -> dict:
                \"\"\"Normalize dictionary keys to lowercase with underscores.

                Args:
                    data: Dictionary with potentially inconsistent keys.

                Returns:
                    Dictionary with normalized keys.
                \"\"\"
                normalized = {{}}
                for key, value in data.items():
                    norm_key = key.lower().replace("-", "_").replace(" ", "_")
                    normalized[norm_key] = value
                return normalized

            async def _chunk_list(self, items: list, chunk_size: int = 10) -> list[list]:
                \"\"\"Split a list into chunks of specified size.

                Args:
                    items: List to split.
                    chunk_size: Maximum items per chunk.

                Returns:
                    List of chunks (sublists).
                \"\"\"
                chunks = []
                for i in range(0, len(items), chunk_size):
                    chunks.append(items[i:i + chunk_size])
                return chunks

            async def _process_in_batches(self, items: list[str]) -> list[dict]:
                \"\"\"Process items in batches with concurrency control.

                Splits items into batches and processes each batch concurrently
                using asyncio.gather while maintaining batch boundaries.

                Args:
                    items: List of item IDs to process.

                Returns:
                    Aggregated list of results from all batches.
                \"\"\"
                chunks = await self._chunk_list(items, self.batch_size)
                all_results = []
                for chunk in chunks:
                    batch_results = await self._batch_process(chunk)
                    all_results.extend(batch_results)
                return all_results

            async def _health_check(self) -> dict:
                \"\"\"Perform a health check against the external API.

                Returns:
                    Dictionary with status and latency information.
                \"\"\"
                import time as _time
                start = _time.monotonic()
                async with httpx.AsyncClient(timeout=5) as client:
                    try:
                        resp = await client.get("https://api.example.com/v1/health")
                        latency = _time.monotonic() - start
                        return {{
                            "status": "healthy" if resp.status_code == 200 else "degraded",
                            "status_code": resp.status_code,
                            "latency_ms": round(latency * 1000, 2),
                        }}
                    except httpx.ConnectError:
                        return {{"status": "unreachable", "latency_ms": -1}}

            async def _build_index(self, items: list[dict], key_field: str = "id") -> dict:
                \"\"\"Build a lookup index from a list of dictionaries.

                Args:
                    items: List of dictionaries to index.
                    key_field: Field to use as the index key.

                Returns:
                    Dictionary mapping key_field values to their items.
                \"\"\"
                index = {{}}
                for item in items:
                    k = item.get(key_field)
                    if k is not None:
                        index[k] = item
                return index

            async def _diff_datasets(self, old: dict, new: dict) -> dict:
                \"\"\"Compute the difference between two datasets.

                Args:
                    old: Previous version of the dataset.
                    new: Current version of the dataset.

                Returns:
                    Dictionary with added, removed, and modified keys.
                \"\"\"
                old_keys = set(old.keys())
                new_keys = set(new.keys())
                added = {{k: new[k] for k in new_keys - old_keys}}
                removed = {{k: old[k] for k in old_keys - new_keys}}
                modified = {{
                    k: {{"old": old[k], "new": new[k]}}
                    for k in old_keys & new_keys
                    if old[k] != new[k]
                }}
                return {{"added": added, "removed": removed, "modified": modified}}

            async def _summarize_batch(self, batch_results: list[dict]) -> dict:
                \"\"\"Summarize a batch of results into aggregate metrics.

                Args:
                    batch_results: List of individual result dictionaries.

                Returns:
                    Summary dictionary with counts and aggregations.
                \"\"\"
                total = len(batch_results)
                success = sum(1 for r in batch_results if r.get("status") == "ok")
                failed = total - success
                return {{
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "success_rate": round(success / max(total, 1) * 100, 2),
                }}

            async def _apply_transformations(
                self, data: dict, transforms: list[str]
            ) -> dict:
                \"\"\"Apply a sequence of named transformations to data.

                Supported transforms: 'uppercase', 'strip', 'sort_keys'.

                Args:
                    data: Input data dictionary.
                    transforms: List of transformation names to apply in order.

                Returns:
                    Transformed data dictionary.
                \"\"\"
                result = dict(data)
                for transform in transforms:
                    if transform == "uppercase":
                        result = {{k: str(v).upper() for k, v in result.items()}}
                    elif transform == "strip":
                        result = {{k.strip(): v for k, v in result.items()}}
                    elif transform == "sort_keys":
                        result = dict(sorted(result.items()))
                return result

            def _run(self, input_data: str) -> dict:
                raise NotImplementedError("Use async version")
    """)


def _issue_blocking_sleep() -> str:
    """Plant issue: time.sleep() in async function."""
    return textwrap.dedent("""\

        class Tool_IssueBlockingSleep(BaseTool):
            name: str = "tool_issue_blocking_sleep"
            description: str = "Tool with a hidden blocking sleep call."

            async def _arun(self, input_data: str) -> dict:
                results = {}
                for i in range(100):
                    results[f"key_{i}"] = f"value_{i}"
                # BUG: blocking sleep in async function
                time.sleep(2)
                return {"processed": len(results), "results": results}

            def _run(self, input_data: str) -> dict:
                raise NotImplementedError("Use async version")
    """)


def _issue_sync_requests() -> str:
    """Plant issue: requests.post() in async function."""
    return textwrap.dedent("""\

        class Tool_IssueSyncRequests(BaseTool):
            name: str = "tool_issue_sync_requests"
            description: str = "Tool with a hidden synchronous HTTP call."

            async def _arun(self, input_data: str) -> dict:
                # BUG: synchronous HTTP call in async function
                response = requests.post(
                    "https://api.example.com/submit",
                    json={"data": input_data},
                    timeout=30,
                )
                return {"status": response.status_code, "body": response.text}

            def _run(self, input_data: str) -> dict:
                raise NotImplementedError("Use async version")
    """)


def _issue_resource_leak() -> str:
    """Plant issue: open() without context manager."""
    return textwrap.dedent("""\

        class Tool_IssueResourceLeak(BaseTool):
            name: str = "tool_issue_resource_leak"
            description: str = "Tool with a resource leak."

            async def _arun(self, file_path: str) -> dict:
                # BUG: open() without context manager — resource leak
                f = open(file_path, "r")
                content = f.read()
                lines = content.split("\\n")
                return {"line_count": len(lines), "char_count": len(content)}

            def _run(self, file_path: str) -> dict:
                raise NotImplementedError("Use async version")
    """)


def _issue_eval_security() -> str:
    """Plant issue: eval() on user input."""
    return textwrap.dedent("""\

        class Tool_IssueEvalSecurity(BaseTool):
            name: str = "tool_issue_eval_security"
            description: str = "Tool with a security vulnerability."

            async def _arun(self, expression: str) -> dict:
                # BUG: eval() on user-controlled input — security vulnerability
                result = eval(expression)
                return {"result": str(result)}

            def _run(self, expression: str) -> dict:
                raise NotImplementedError("Use async version")
    """)


def _issue_cpu_bound_crypto() -> str:
    """Plant issue: hashlib.pbkdf2_hmac in async without to_thread."""
    return textwrap.dedent("""\

        class Tool_IssueCpuBoundCrypto(BaseTool):
            name: str = "tool_issue_cpu_bound_crypto"
            description: str = "Tool with CPU-bound crypto in async context."

            async def _arun(self, password: str) -> dict:
                # BUG: CPU-bound crypto in async function without to_thread
                dk = hashlib.pbkdf2_hmac(
                    "sha256",
                    password.encode(),
                    b"static_salt_value",
                    100000,
                )
                return {"derived_key": dk.hex()}

            def _run(self, password: str) -> dict:
                raise NotImplementedError("Use async version")
    """)


def generate_large_file(output_path: str) -> list[dict]:
    """
    Generate a ~15K LOC file with 5 planted issues at known positions.

    Returns a list of planted issue descriptors with exact line numbers.
    """
    header = textwrap.dedent("""\
        \"\"\"Auto-generated large tool file for benchmark testing.

        This file contains ~15K lines of realistic async tool code with
        5 intentional issues planted at specific locations.
        \"\"\"

        import asyncio
        import hashlib
        import time
        from typing import Optional

        import httpx
        import aiofiles
        import requests
        from langchain.tools import BaseTool

    """)

    parts: list[str] = [header]
    planted_issues: list[dict] = []

    # ~36 padding classes + 5 issue classes to reach ~15K lines.
    # Insert issues at class positions: 1, 7, 18, 27, 35
    issue_insertions = {
        1: ("blocking-call", "time.sleep(2) in async method", _issue_blocking_sleep),
        7: ("sync_code", "requests.post() in async method", _issue_sync_requests),
        18: ("memory_leak", "open() without context manager", _issue_resource_leak),
        27: ("security", "eval() on user input", _issue_eval_security),
        35: ("crypto", "hashlib.pbkdf2_hmac in async without to_thread", _issue_cpu_bound_crypto),
    }

    class_index = 0
    for i in range(37):
        if i in issue_insertions:
            category, description, generator = issue_insertions[i]
            issue_code = generator()
            current_line = sum(part.count("\n") for part in parts) + 1
            # Find the actual bug line within the issue snippet
            for offset, line in enumerate(issue_code.split("\n")):
                if "BUG:" in line:
                    bug_line = current_line + offset
                    break
            else:
                bug_line = current_line + 5
            planted_issues.append({
                "line": bug_line,
                "category": category,
                "description": description,
            })
            parts.append(issue_code)
        else:
            parts.append(_padding_class(class_index))
            class_index += 1

    full_content = "".join(parts)

    with open(output_path, "w") as f:
        f.write(full_content)

    total_lines = full_content.count("\n") + 1
    print(f"  Generated {output_path}: {total_lines} lines, {len(planted_issues)} planted issues")
    for issue in planted_issues:
        print(f"    - Line {issue['line']}: [{issue['category']}] {issue['description']}")

    return planted_issues


if __name__ == "__main__":
    issues = generate_large_file("samples/large_bad.py")
    print(f"\nPlanted {len(issues)} issues")
