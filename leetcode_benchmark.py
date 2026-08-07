#!/usr/bin/env python3
"""
leetcode_benchmark.py — LeetCode-style coding accuracy benchmark

Methodology follows the coding-ability test design at https://klab.tw/2026/06/gemma4-benchmark/:
  1. Send the problem (including the required function signature) to the model,
     asking only for code output
  2. Extract the code block from the response
  3. Assemble the code + test cases into a full test script, run it inside an
     "isolated Docker container"
  4. A problem only counts as passed if all test cases succeed; results are
     tallied separately by difficulty (easy/medium/hard)

Prerequisite: Docker must be available on the machine, and able to run
`docker run python:3.12-slim` (the image is pulled automatically on first run).

Usage:
    python3 leetcode_benchmark.py --models gemma4:26b-64k gemma4:31b-64k \
        --problems sample_leetcode_problems.json --output leetcode_results
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
import uuid
import urllib.request
import urllib.error
from pathlib import Path


# ── Check whether Docker is available ─────────────────────

def check_docker_available():
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Call Ollama to generate code ──────────────────────────

def call_ollama_generate(base_url, model, prompt, num_ctx=32768, timeout=600, think=None):
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }
    if think is not None:
        body["think"] = think
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", "")


def build_coding_prompt(problem):
    return (
        f"Implement the following function in Python. Output only the code "
        f"(wrapped in ```python), no explanatory text. The function name and "
        f"parameters must exactly match the given signature; do not change them.\n\n"
        f"Problem: {problem['title']}\n"
        f"{problem['description']}\n\n"
        f"Function signature: {problem['signature']}"
    )


def extract_code_block(response_text):
    """Extracts the first ```python ... ``` or ``` ... ``` code block from the model's response."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If there's no code fence, assume the whole response is code
    # (some small models don't add fences)
    return response_text.strip()


# ── Run test cases inside a Docker sandbox ────────────────

def build_test_harness(code, function_name, test_cases):
    """
    Assembles the model-generated code + test cases into a complete Python test
    script. Each test case's input/expected output is parsed with
    ast.literal_eval, to avoid the risks of using eval directly.
    """
    lines = [
        code,
        "",
        "import sys, ast",
        f"_fn = {function_name}",
        "_results = []",
    ]
    for i, tc in enumerate(test_cases):
        lines.append(f"try:")
        lines.append(f"    _args = ast.literal_eval({tc['input']!r})")
        lines.append(f"    _expected = ast.literal_eval({tc['expected']!r})")
        lines.append(f"    _actual = _fn(*_args)")
        lines.append(f"    _results.append(_actual == _expected)")
        lines.append(f"except Exception as e:")
        lines.append(f"    _results.append(False)")
    lines.append("print('RESULTS:' + ','.join('1' if r else '0' for r in _results))")
    return "\n".join(lines)


def run_in_docker(script_content, timeout=15, memory="256m"):
    """
    Writes the test script to a temp file and runs it inside an isolated Docker
    container (--network none blocks network access, to prevent generated code
    from making unexpected network calls; --rm automatically cleans up the
    container after it finishes).
    Returns (success, stdout, stderr).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_path.write_text(script_content, encoding="utf-8")

        container_name = f"leetcode-bench-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network", "none",
            f"--memory={memory}",
            "--cpus=1",
            "-v", f"{tmpdir}:/app:ro",
            "python:3.12-slim",
            "python", "/app/test_script.py",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            return True, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            # If it times out, the container may still be running; force-kill it
            # to avoid leaving it behind
            subprocess.run(["docker", "kill", container_name],
                            capture_output=True, timeout=10)
            return False, "", "Execution timed out (may be an infinite loop or very poor performance)"


def grade_problem(base_url, model, problem, timeout=600, docker_timeout=15, think=None):
    prompt = build_coding_prompt(problem)
    gen_start = time.perf_counter()
    try:
        response = call_ollama_generate(base_url, model, prompt, timeout=timeout, think=think)
    except (urllib.error.URLError, TimeoutError) as e:
        return {"passed": False, "error": f"Model call failed: {e}", "test_pass_count": 0,
                "test_total": len(problem["test_cases"]), "response_chars": 0,
                "gen_duration_s": round(time.perf_counter() - gen_start, 2)}
    gen_duration = time.perf_counter() - gen_start

    code = extract_code_block(response)
    harness = build_test_harness(code, problem["function_name"], problem["test_cases"])
    ok, stdout, stderr = run_in_docker(harness, timeout=docker_timeout)

    if not ok:
        return {"passed": False, "error": stderr, "test_pass_count": 0,
                "test_total": len(problem["test_cases"]), "raw_code": code,
                "response_chars": len(response), "gen_duration_s": round(gen_duration, 2)}

    match = re.search(r"RESULTS:([01,]+)", stdout)
    if not match:
        return {"passed": False, "error": f"Could not parse execution results: stdout={stdout!r} stderr={stderr!r}",
                "test_pass_count": 0, "test_total": len(problem["test_cases"]), "raw_code": code,
                "response_chars": len(response), "gen_duration_s": round(gen_duration, 2)}

    results = [c == "1" for c in match.group(1).split(",")]
    passed = all(results)
    return {
        "passed": passed,
        "test_pass_count": sum(results),
        "test_total": len(results),
        "raw_code": code,
        "response_chars": len(response),
        "gen_duration_s": round(gen_duration, 2),
    }


# ── Main flow ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeetCode-style coding accuracy benchmark (runs in a Docker sandbox)")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--problems", required=True, help="Path to the problem bank JSON file")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=600, help="Timeout limit for model generation, in seconds")
    parser.add_argument("--docker-timeout", type=int, default=15, help="Timeout limit for code execution per problem, in seconds")
    parser.add_argument("--think", action="store_true", default=None,
                         help="Enable thinking mode (only effective for models that support reasoning)")
    parser.add_argument("--no-think", dest="think", action="store_false",
                         help="Disable thinking mode")
    parser.add_argument("--output", default="leetcode_results")
    args = parser.parse_args()

    if not check_docker_available():
        print("[ERROR] No usable Docker detected (`docker info` failed).\n"
              "This script requires Docker to run model-generated code in isolation, "
              "protecting the host from problematic code.\n"
              "Install and start Docker first: sudo systemctl start docker, "
              "or confirm the current user has permission to run docker commands "
              "(may require sudo, or adding the user to the docker group).",
              file=sys.stderr)
        sys.exit(1)

    with open(args.problems, "r", encoding="utf-8") as f:
        problems = json.load(f)
    print(f"Loaded problem bank: {len(problems)} problems", file=sys.stderr)

    all_results = []
    summary_rows = []

    for model in args.models:
        print(f"\n{'='*60}\nStarting test for model: {model}\n{'='*60}", file=sys.stderr)
        by_difficulty = {}
        for p in problems:
            diff = p.get("difficulty", "unknown")
            by_difficulty.setdefault(diff, {"total": 0, "passed": 0})

        for idx, p in enumerate(problems, 1):
            print(f"  [{model}] Problem {idx}/{len(problems)}: {p['title']} ({p.get('difficulty', '?')})...",
                  file=sys.stderr, end=" ")
            start = time.perf_counter()
            result = grade_problem(args.url, model, p, timeout=args.timeout,
                                    docker_timeout=args.docker_timeout, think=args.think)
            elapsed = time.perf_counter() - start

            diff = p.get("difficulty", "unknown")
            by_difficulty[diff]["total"] += 1
            if result["passed"]:
                by_difficulty[diff]["passed"] += 1
                print(f"PASSED ({elapsed:.1f}s, response {result.get('response_chars', 0)} chars)", file=sys.stderr)
            else:
                print(f"FAILED ({elapsed:.1f}s): {result.get('error', 'not all test cases passed')}", file=sys.stderr)

            all_results.append({
                "model": model,
                "think_mode": args.think if args.think is not None else "model_default",
                "problem_id": p.get("id", idx),
                "title": p["title"],
                "difficulty": diff,
                "passed": result["passed"],
                "test_pass_count": result["test_pass_count"],
                "test_total": result["test_total"],
                "elapsed_s": round(elapsed, 2),
                "gen_duration_s": result.get("gen_duration_s", 0),
                "response_chars": result.get("response_chars", 0),
                "error": result.get("error", ""),
            })

        print(f"\n--- {model} results by difficulty ---", file=sys.stderr)
        total_passed, total_count = 0, 0
        model_results = [r for r in all_results if r["model"] == model]
        avg_gen_duration = (sum(r["gen_duration_s"] for r in model_results) / len(model_results)
                             if model_results else 0)
        for diff, stats in by_difficulty.items():
            rate = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
            print(f"  {diff}: {stats['passed']}/{stats['total']} ({rate:.1f}%)", file=sys.stderr)
            total_passed += stats["passed"]
            total_count += stats["total"]
            summary_rows.append({
                "model": model,
                "think_mode": args.think if args.think is not None else "model_default",
                "difficulty": diff,
                "passed": stats["passed"], "total": stats["total"],
                "pass_rate_pct": round(rate, 1),
                "avg_gen_duration_s": round(avg_gen_duration, 2),
            })
        overall_rate = total_passed / total_count * 100 if total_count else 0
        print(f"  Overall: {total_passed}/{total_count} ({overall_rate:.1f}%), "
              f"average generation time {avg_gen_duration:.1f}s", file=sys.stderr)
        summary_rows.append({
            "model": model,
            "think_mode": args.think if args.think is not None else "model_default",
            "difficulty": "overall",
            "passed": total_passed, "total": total_count,
            "pass_rate_pct": round(overall_rate, 1),
            "avg_gen_duration_s": round(avg_gen_duration, 2),
        })

    # Save results (filename automatically includes the think mode, to avoid
    # think/no-think runs overwriting each other's results)
    think_suffix = {"True": "think", "False": "nothink", "None": "default"}[
        str(args.think)
    ]
    output_prefix = f"{args.output}_{think_suffix}"

    import csv
    with open(f"{output_prefix}_detail.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    with open(f"{output_prefix}_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n\nDetail results saved to {output_prefix}_detail.csv", file=sys.stderr)
    print(f"Summary saved to {output_prefix}_summary.csv", file=sys.stderr)

    print(f"\n===== Final Summary (think={args.think}) =====")
    header = f"{'Model':<25} {'Difficulty':<10} {'Passed':>6} {'Total':>6} {'Pass rate (%)':>14} {'Avg gen time (s)':>18}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(f"{row['model']:<25} {row['difficulty']:<10} {row['passed']:>6} "
              f"{row['total']:>6} {row['pass_rate_pct']:>14} {row['avg_gen_duration_s']:>18}")


if __name__ == "__main__":
    main()
