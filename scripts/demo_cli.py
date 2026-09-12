"""Week 4 demo: type a query, watch the cascade router's live decision.

Run: ./venv/bin/python3 scripts/demo_cli.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.cascade import run_cascade

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"

TIER_COLOR = {"cheap": GREEN, "mid": CYAN, "frontier": YELLOW}
TIER_ORDER = ["cheap", "mid", "frontier"]

BANNER = f"""
{BOLD}LLM Router -- live demo{RESET}
{DIM}Type a query and watch it get classified, routed, and (if needed) escalated.
Type 'quit' to exit.{RESET}
"""


def format_router_line(result: dict) -> str:
    tier = result["final_tier"]
    color = TIER_COLOR.get(tier, RESET)
    base = (f"{color}Routed to: {tier}{RESET} "
            f"{DIM}(classified {result['difficulty']}){RESET} · "
            f"${result['total_cost']:.5f} · {result['total_latency_ms']:.0f}ms")

    if result["escalated"]:
        # The exact tiers attempted are fully determined by initial_tier and
        # final_tier (the cascade only ever moves up TIER_ORDER one step at a
        # time) -- reconstructing it this way is exact, unlike parsing the
        # free-text escalation_reasons strings.
        start_i = TIER_ORDER.index(result["initial_tier"])
        end_i = TIER_ORDER.index(tier)
        chain = " -> ".join(TIER_ORDER[start_i:end_i + 1])
        base += f"\n{RED}Escalated{RESET} ({chain})"
        for reason in result["escalation_reasons"]:
            base += f"\n  {DIM}- {reason[:100]}{RESET}"
    return base


def main():
    print(BANNER)
    while True:
        try:
            query = input(f"{BOLD}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        start = time.perf_counter()
        try:
            result = run_cascade([{"role": "user", "content": query}])
        except Exception as exc:
            print(f"{RED}Error: {exc}{RESET}\n")
            continue
        elapsed = time.perf_counter() - start

        print(f"\n{result['text']}\n")
        print(format_router_line(result))
        print(f"{DIM}(wall time: {elapsed:.1f}s including any escalation){RESET}\n")


if __name__ == "__main__":
    main()
