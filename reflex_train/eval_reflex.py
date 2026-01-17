import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@dataclass
class Summary:
    total_frames: int = 0
    intent_counts: Counter = None
    pressed_counts: Counter = None
    released_counts: Counter = None
    active_counts: Counter = None
    active_total: int = 0
    event_frames: int = 0
    run_lengths: dict = None
    intent_transitions: int = 0
    last_intent: str | None = None
    per_key_runs: dict = None
    max_simul_active: int = 0


def summarize(records):
    summary = Summary(
        intent_counts=Counter(),
        pressed_counts=Counter(),
        released_counts=Counter(),
        active_counts=Counter(),
        run_lengths=defaultdict(list),
        per_key_runs=defaultdict(list),
    )

    active_run_lengths = defaultdict(int)

    for rec in records:
        summary.total_frames += 1
        intent = rec.get("pred_intent") or "<none>"
        summary.intent_counts[intent] += 1

        if summary.last_intent is not None and intent != summary.last_intent:
            summary.intent_transitions += 1
        summary.last_intent = intent

        pressed = rec.get("pressed") or []
        released = rec.get("released") or []
        active = rec.get("active") or []

        summary.pressed_counts.update(pressed)
        summary.released_counts.update(released)
        summary.active_counts.update(active)
        summary.active_total += len(active)
        summary.max_simul_active = max(summary.max_simul_active, len(active))

        if pressed or released:
            summary.event_frames += 1

        active_set = set(active)
        for key in list(active_run_lengths.keys()):
            if key not in active_set:
                summary.per_key_runs[key].append(active_run_lengths[key])
                active_run_lengths[key] = 0

        for key in active_set:
            active_run_lengths[key] += 1

    for key, run_len in active_run_lengths.items():
        if run_len:
            summary.per_key_runs[key].append(run_len)

    for key, runs in summary.per_key_runs.items():
        if runs:
            summary.run_lengths[key] = {
                "count": len(runs),
                "mean": sum(runs) / len(runs),
                "max": max(runs),
            }

    return summary


def print_top(counter, limit):
    for key, count in counter.most_common(limit):
        print(f"  {key}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reflex_infer predictions JSONL."
    )
    parser.add_argument("preds", help="Path to preds.jsonl")
    parser.add_argument("--top", type=int, default=8, help="Top N keys/intents to show")
    args = parser.parse_args()

    records = load_jsonl(args.preds)
    if not records:
        print("No records found.")
        return

    summary = summarize(records)
    total = summary.total_frames
    active_rate = summary.active_total / total if total else 0
    event_rate = summary.event_frames / total if total else 0

    print("Summary")
    print(f"  frames: {summary.total_frames}")
    print(f"  intent transitions: {summary.intent_transitions}")
    print(f"  event frames: {summary.event_frames} ({event_rate:.2%})")
    print(f"  mean active keys/frame: {active_rate:.2f}")
    print(f"  max simultaneous active keys: {summary.max_simul_active}")

    print("\nIntent distribution")
    print_top(summary.intent_counts, args.top)

    print("\nTop pressed keys")
    print_top(summary.pressed_counts, args.top)

    print("\nTop released keys")
    print_top(summary.released_counts, args.top)

    print("\nTop active keys")
    print_top(summary.active_counts, args.top)

    if summary.run_lengths:
        print("\nActive run stats")
        for key, stats in sorted(
            summary.run_lengths.items(),
            key=lambda item: item[1]["max"],
            reverse=True,
        )[: args.top]:
            print(
                f"  {key}: count={stats['count']} mean={stats['mean']:.2f} max={stats['max']}"
            )


if __name__ == "__main__":
    main()
