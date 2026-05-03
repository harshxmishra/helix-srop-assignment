"""
E7: Routing accuracy eval harness.

Usage:
    python eval/run_eval.py [--base-url http://localhost:8000]

Runs a fixed set of labeled queries against the running server.
Reports routing accuracy (actual routed_to == expected).
Prints per-query results and summary.
"""
import argparse
import asyncio
import sys

import httpx

# Labeled eval dataset: (query, expected_routed_to)
EVAL_CASES = [
    # Knowledge queries
    ("How do I rotate a deploy key?",               "knowledge"),
    ("What is the difference between free and pro?", "knowledge"),
    ("How do I configure secret scanning?",          "knowledge"),
    ("What webhook events are available?",           "knowledge"),
    ("How do I set up a custom runner?",             "knowledge"),
    ("What is the artifact registry used for?",     "knowledge"),
    ("How do I use observability in Helix CI?",     "knowledge"),
    # Account queries
    ("Show me my recent builds",                     "account"),
    ("What is my account status?",                   "account"),
    ("How many concurrent builds do I have left?",   "account"),
    # Escalation
    ("I need to talk to a human",                    "escalation_agent"),
    ("Please escalate my issue",                     "escalation_agent"),
    # Guardrails
    ("Write me a poem",                              "guardrails"),
    ("Tell me a joke",                               "guardrails"),
]


async def run_eval(base_url: str) -> None:
    user_id = "eval_user_001"

    async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
        # Create eval session
        resp = await client.post("/v1/sessions", json={"user_id": user_id, "plan_tier": "pro"})
        if resp.status_code != 200:
            print(f"ERROR: Could not create session: {resp.text}")
            sys.exit(1)
        session_id = resp.json()["session_id"]
        print(f"Eval session: {session_id}\n")

        results = []
        for query, expected in EVAL_CASES:
            resp = await client.post(
                f"/v1/chat/{session_id}",
                json={"content": query},
            )
            if resp.status_code != 200:
                print(f"  FAIL [{resp.status_code}] {query!r}")
                results.append(False)
                continue

            data = resp.json()
            actual = data.get("routed_to", "?")
            correct = actual == expected
            results.append(correct)

            status = "✓" if correct else "✗"
            print(f"  {status} expected={expected:<20} actual={actual:<20} | {query}")

        total = len(results)
        passed = sum(results)
        accuracy = passed / total * 100
        print(f"\nRouting accuracy: {passed}/{total} = {accuracy:.1f}%")

        if accuracy < 70:
            print("WARNING: Accuracy below 70% — review routing instructions.")
        else:
            print("PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Routing accuracy eval")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    asyncio.run(run_eval(args.base_url))


if __name__ == "__main__":
    main()
