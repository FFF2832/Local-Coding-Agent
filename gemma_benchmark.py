#!/usr/bin/env python3
"""
gemma_benchmark.py — Three-metric benchmark script for local Ollama models
Testing methodology follows the design of https://klab.tw/2026/06/gemma4-benchmark/:
  - Throughput (decode speed): eval_count / eval_duration
  - Latency: prefill time (proxy for time-to-first-token) + total time per question
  - Accuracy: automatically graded against a fixed question bank, no manual comparison needed

Depends only on the Python standard library, no pip install required, so it can run
directly on the deployment machine.

Usage examples:
    # Speed test only (Latency + Throughput), comparing two models
    python3 gemma_benchmark.py speed --models gemma4:26b-32k gemma4:31b

    # Accuracy test, requires a question bank JSON (see sample_questions.json for format)
    python3 gemma_benchmark.py accuracy --models gemma4:26b-32k gemma4:31b \
        --questions sample_questions.json

    # Run both, saved to CSV
    python3 gemma_benchmark.py all --models gemma4:26b-32k gemma4:31b \
        --questions sample_questions.json --output results.csv
"""

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.request
import urllib.error


# ── Ollama API calls ─────────────────────────────────────

def call_ollama_generate(base_url, model, prompt, num_ctx=32768,
                          num_predict=None, think=None, timeout=900):
    """
    Calls Ollama's /api/generate with stream=False to get the full response at once.
    Returns a dict containing the response text and timing fields (all in nanoseconds,
    convert as needed). Default timeout is 900 seconds (15 minutes), because models
    that exceed VRAM capacity and need to offload to system RAM (e.g. a 33GB model on
    a 32GB VRAM card) may take far longer than usual for a single generation.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }
    if num_predict is not None:
        body["options"]["num_predict"] = num_predict
    if think is not None:
        body["think"] = think

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except TimeoutError as e:
        raise RuntimeError(
            f"Timeout (no response after {timeout} seconds): {model} took too long to "
            f"generate. This usually means the model size exceeds VRAM capacity and had "
            f"to offload to system RAM, causing a major slowdown. You can raise the limit "
            f"with --timeout, or treat this as evidence the model isn't viable on this hardware."
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Ollama ({base_url}): {e}") from e

    result["_wall_clock_seconds"] = time.perf_counter() - start
    return result


def ns_to_s(nanoseconds):
    """Ollama's duration fields are returned in nanoseconds; convert to seconds."""
    return nanoseconds / 1e9 if nanoseconds else 0.0


# ── Speed test (Latency + Throughput) ─────────────────────

# Several different prompt types, covering long-form writing, algorithm
# implementation, sequences, and debugging — common task categories — to
# avoid a single prompt skewing the speed numbers toward one task's
# characteristics.
SPEED_TEST_PROMPTS = [
    {
        "id": "essay",
        "label": "Long-form writing",
        "prompt": "Write a short essay of at least 400 words on the topic \"Pros and cons of deploying large language models locally.\"",
    },
    {
        "id": "fibonacci",
        "label": "Fibonacci sequence",
        "prompt": "Write a Python function that computes the Nth term of the Fibonacci sequence, with a brief explanation.",
    },
    {
        "id": "sort_algo",
        "label": "Sorting algorithm",
        "prompt": "Implement quicksort in Python and explain its time complexity.",
    },
    {
        "id": "debug",
        "label": "Debugging task",
        "prompt": (
            "The following Python code has a bug. Find and fix it:\n\n"
            "def divide_list(numbers, divisor):\n"
            "    result = []\n"
            "    for n in numbers:\n"
            "        result.append(n / divisor)\n"
            "    return result\n\n"
            "print(divide_list([1, 2, 3], 0))"
        ),
    },
]


def run_speed_test(base_url, model, num_ctx, num_runs, think=None, prompts=None, timeout=900):
    """
    Runs the speed test for a single model: each prompt group runs num_runs times.
    Returns the overall average, per-prompt averages, and the raw data for every run.
    A single failed call (timeout/connection error) does not abort the whole test —
    the error is recorded, that sample is skipped, and testing continues with the
    next run. Final averages only include successful samples.
    """
    if prompts is None:
        prompts = SPEED_TEST_PROMPTS

    runs = []
    failed = []
    for p in prompts:
        for i in range(num_runs):
            print(f"  [{model}] [{p['label']}] Run {i + 1}/{num_runs}...", file=sys.stderr)
            try:
                result = call_ollama_generate(
                    base_url, model, p["prompt"],
                    num_ctx=num_ctx, think=think, timeout=timeout,
                )
            except RuntimeError as e:
                print(f"    Failed, skipping this sample: {e}", file=sys.stderr)
                failed.append({
                    "prompt_id": p["id"], "prompt_label": p["label"],
                    "run": i + 1, "error": str(e),
                })
                continue

            prompt_eval_count = result.get("prompt_eval_count", 0)
            prompt_eval_duration = ns_to_s(result.get("prompt_eval_duration", 0))
            eval_count = result.get("eval_count", 0)
            eval_duration = ns_to_s(result.get("eval_duration", 0))
            total_duration = ns_to_s(result.get("total_duration", 0))

            decode_tps = eval_count / eval_duration if eval_duration > 0 else 0
            prefill_tps = prompt_eval_count / prompt_eval_duration if prompt_eval_duration > 0 else 0

            runs.append({
                "prompt_id": p["id"],
                "prompt_label": p["label"],
                "run": i + 1,
                "prompt_eval_count": prompt_eval_count,
                "prompt_eval_duration_s": round(prompt_eval_duration, 4),
                "eval_count": eval_count,
                "eval_duration_s": round(eval_duration, 4),
                "total_duration_s": round(total_duration, 4),
                "decode_tok_s": round(decode_tps, 2),
                "prefill_tok_s": round(prefill_tps, 2),
            })

    if not runs:
        # All samples failed (e.g. wrong model name, or completely unreachable) —
        # return an empty result for the caller to handle
        summary = {
            "model": model, "total_samples": 0, "failed_samples": len(failed),
            "avg_decode_tok_s": None, "avg_prefill_tok_s": None,
            "avg_total_latency_s": None, "avg_prefill_latency_s": None,
        }
        return summary, [], runs, failed

    # Overall average (all prompts x all runs combined, successful samples only)
    decode_vals = [r["decode_tok_s"] for r in runs]
    prefill_vals = [r["prefill_tok_s"] for r in runs]
    total_vals = [r["total_duration_s"] for r in runs]
    prefill_dur_vals = [r["prompt_eval_duration_s"] for r in runs]

    summary = {
        "model": model,
        "total_samples": len(runs),
        "failed_samples": len(failed),
        "avg_decode_tok_s": round(statistics.mean(decode_vals), 2),
        "avg_prefill_tok_s": round(statistics.mean(prefill_vals), 2),
        "avg_total_latency_s": round(statistics.mean(total_vals), 2),
        "avg_prefill_latency_s": round(statistics.mean(prefill_dur_vals), 4),
    }

    # Per-prompt-type averages (successful samples only), useful for seeing
    # speed differences across task types
    per_prompt_summary = []
    for p in prompts:
        p_runs = [r for r in runs if r["prompt_id"] == p["id"]]
        if not p_runs:
            per_prompt_summary.append({
                "model": model, "prompt_id": p["id"], "prompt_label": p["label"],
                "avg_decode_tok_s": None, "avg_prefill_tok_s": None, "avg_total_latency_s": None,
            })
            continue
        per_prompt_summary.append({
            "model": model,
            "prompt_id": p["id"],
            "prompt_label": p["label"],
            "avg_decode_tok_s": round(statistics.mean([r["decode_tok_s"] for r in p_runs]), 2),
            "avg_prefill_tok_s": round(statistics.mean([r["prefill_tok_s"] for r in p_runs]), 2),
            "avg_total_latency_s": round(statistics.mean([r["total_duration_s"] for r in p_runs]), 2),
        })

    return summary, per_prompt_summary, runs, failed


# ── Accuracy test ──────────────────────────────────────────

def build_mc_prompt(question):
    """Formats a single multiple-choice question into a fixed prompt structure, for easy parsing."""
    lines = [f"The following is a multiple-choice question. Answer with only the option letter (e.g. A), no explanation or extra text.", ""]
    lines.append(question["question"])
    for key in sorted(question["options"].keys()):
        lines.append(f"{key}. {question['options'][key]}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def extract_answer_letter(response_text, valid_letters):
    """Extracts the first valid option letter found in the model's response."""
    for ch in response_text.strip():
        upper = ch.upper()
        if upper in valid_letters:
            return upper
    return None


def run_accuracy_test(base_url, model, questions, num_ctx, think=None, num_predict=None):
    correct = 0
    details = []
    total = len(questions)

    for idx, q in enumerate(questions, start=1):
        print(f"  [{model}] Question {idx}/{total}...", file=sys.stderr)
        prompt = build_mc_prompt(q)
        try:
            result = call_ollama_generate(
                base_url, model, prompt,
                num_ctx=num_ctx, num_predict=num_predict, think=think,
            )
        except RuntimeError as e:
            print(f"    Error: {e}", file=sys.stderr)
            details.append({"id": q.get("id", idx), "correct": False, "error": str(e)})
            continue

        response_text = result.get("response", "")
        valid_letters = set(q["options"].keys())
        answer = extract_answer_letter(response_text, valid_letters)
        is_correct = (answer == q["answer"])
        if is_correct:
            correct += 1

        details.append({
            "id": q.get("id", idx),
            "model_answer": answer,
            "correct_answer": q["answer"],
            "correct": is_correct,
            "raw_response": response_text[:200],
        })

    accuracy = correct / total if total > 0 else 0
    summary = {
        "model": model,
        "total_questions": total,
        "correct": correct,
        "accuracy": round(accuracy * 100, 2),
    }
    return summary, details


# ── Output ───────────────────────────────────────────────

def print_speed_table(summaries):
    print("\n===== Speed Test Results (Latency + Throughput) =====")
    header = f"{'Model':<20} {'decode tok/s':>14} {'prefill tok/s':>14} {'Total (s)':>10} {'Prefill latency (s)':>20} {'Failed':>8}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        if s.get("avg_decode_tok_s") is None:
            print(f"{s['model']:<20} {'ALL FAILED':>14} {'-':>14} {'-':>10} {'-':>20} {s.get('failed_samples', '?'):>8}")
            continue
        print(f"{s['model']:<20} {s['avg_decode_tok_s']:>14} {s['avg_prefill_tok_s']:>14} "
              f"{s['avg_total_latency_s']:>10} {s['avg_prefill_latency_s']:>20} {s.get('failed_samples', 0):>8}")


def print_speed_per_prompt_table(per_prompt_summaries):
    print("\n===== Speed Test Results (by task type) =====")
    header = f"{'Model':<20} {'Task type':<20} {'decode tok/s':>14} {'prefill tok/s':>14} {'Total (s)':>10}"
    print(header)
    print("-" * len(header))
    for s in per_prompt_summaries:
        if s.get("avg_decode_tok_s") is None:
            print(f"{s['model']:<20} {s['prompt_label']:<20} {'FAILED':>14} {'-':>14} {'-':>10}")
            continue
        print(f"{s['model']:<20} {s['prompt_label']:<20} {s['avg_decode_tok_s']:>14} "
              f"{s['avg_prefill_tok_s']:>14} {s['avg_total_latency_s']:>10}")


def print_accuracy_table(summaries):
    print("\n===== Accuracy Test Results =====")
    header = f"{'Model':<20} {'Questions':>10} {'Correct':>10} {'Accuracy (%)':>14}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(f"{s['model']:<20} {s['total_questions']:>10} {s['correct']:>10} {s['accuracy']:>14}")


def save_csv(rows, path, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {path}", file=sys.stderr)


# ── Main flow ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gemma4 local model three-metric benchmark script")
    parser.add_argument("mode", choices=["speed", "accuracy", "all"],
                         help="Which test to run: speed (latency+throughput) / accuracy / all (both)")
    parser.add_argument("--models", nargs="+", required=True,
                         help="Model tags to test, e.g. gemma4:26b-32k gemma4:31b")
    parser.add_argument("--url", default="http://127.0.0.1:11434",
                         help="Ollama API address, defaults to http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=32768,
                         help="Context length, defaults to 32768")
    parser.add_argument("--num-runs", type=int, default=3,
                         help="Number of repetitions per test prompt, defaults to 3 runs averaged"
                              " (the speed test has 4 different task-type prompts, so total samples = 4 x num-runs)")
    parser.add_argument("--questions", default=None,
                         help="Path to the accuracy test's question bank JSON file (see sample_questions.json for format)")
    parser.add_argument("--think", action="store_true", default=None,
                         help="Enable thinking mode (only effective for models that support reasoning)")
    parser.add_argument("--no-think", dest="think", action="store_false",
                         help="Disable thinking mode")
    parser.add_argument("--num-predict", type=int, default=None,
                         help="Cap the max output tokens during the accuracy test (avoids thinking mode running too long)")
    parser.add_argument("--timeout", type=int, default=900,
                         help="Timeout limit per API call, in seconds, defaults to 900 (15 minutes)."
                              " May need to be raised when testing large models near or over VRAM capacity")
    parser.add_argument("--output", default=None,
                         help="Path prefix for saving results as CSV, e.g. 'results' produces results_speed.csv, etc.")

    args = parser.parse_args()

    if args.mode in ("accuracy", "all") and not args.questions:
        parser.error("accuracy / all mode requires --questions to specify the question bank JSON path")

    speed_summaries = []
    speed_per_prompt_summaries = []
    accuracy_summaries = []

    if args.mode in ("speed", "all"):
        for model in args.models:
            print(f"\nStarting speed test for {model}...", file=sys.stderr)
            summary, per_prompt_summary, runs, failed = run_speed_test(
                args.url, model, args.num_ctx, args.num_runs,
                think=args.think, timeout=args.timeout,
            )
            speed_summaries.append(summary)
            speed_per_prompt_summaries.extend(per_prompt_summary)
            if failed:
                print(f"  {model} had {len(failed)} failed sample(s) (timeout or connection issue), "
                      f"averages only include the {summary['total_samples']} successful sample(s)", file=sys.stderr)
            if args.output and runs:
                save_csv(runs, f"{args.output}_speed_{model.replace(':', '_')}.csv",
                          fieldnames=list(runs[0].keys()))
            if args.output and failed:
                save_csv(failed, f"{args.output}_speed_{model.replace(':', '_')}_failed.csv",
                          fieldnames=list(failed[0].keys()))
        print_speed_table(speed_summaries)
        print_speed_per_prompt_table(speed_per_prompt_summaries)
        if args.output and speed_per_prompt_summaries:
            save_csv(speed_per_prompt_summaries, f"{args.output}_speed_per_prompt.csv",
                      fieldnames=list(speed_per_prompt_summaries[0].keys()))

    if args.mode in ("accuracy", "all"):
        with open(args.questions, "r", encoding="utf-8") as f:
            questions = json.load(f)
        print(f"\nLoaded question bank: {len(questions)} questions", file=sys.stderr)

        for model in args.models:
            print(f"\nStarting accuracy test for {model}...", file=sys.stderr)
            summary, details = run_accuracy_test(
                args.url, model, questions, args.num_ctx,
                think=args.think, num_predict=args.num_predict,
            )
            accuracy_summaries.append(summary)
            if args.output:
                save_csv(details, f"{args.output}_accuracy_{model.replace(':', '_')}.csv",
                          fieldnames=list(details[0].keys()))
        print_accuracy_table(accuracy_summaries)

    if args.output:
        all_rows = []
        for s in speed_summaries:
            row = dict(s)
            row["metric_type"] = "speed"
            all_rows.append(row)
        for s in accuracy_summaries:
            row = dict(s)
            row["metric_type"] = "accuracy"
            all_rows.append(row)
        if all_rows:
            fieldnames = sorted({k for row in all_rows for k in row.keys()})
            save_csv(all_rows, f"{args.output}_summary.csv", fieldnames=fieldnames)


if __name__ == "__main__":
    main()
