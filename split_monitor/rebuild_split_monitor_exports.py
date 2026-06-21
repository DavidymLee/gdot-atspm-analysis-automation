from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose_mode(values: list[str]) -> str:
    counts = Counter(values)
    max_count = max(counts.values())
    candidates = {value for value, count in counts.items() if count == max_count}
    for value in values:
        if value in candidates:
            return value
    return values[0]


def rebuild_hourly_from_fulltrace(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        minute = int(row["Time of day"])
        hour = min(23, minute // 60)
        key = (row["Signal ID"], row["Phase #"], row["Date"], hour)
        grouped[key].append(row)

    rebuilt_rows: list[dict[str, str]] = []
    for signal_id, phase, date_text, hour in sorted(grouped.keys(), key=lambda x: (int(x[1]), x[3])):
        group = grouped[(signal_id, phase, date_text, hour)]
        rebuilt_rows.append(
            {
                "Signal ID": signal_id,
                "Phase #": phase,
                "Date": date_text,
                "Plan": choose_mode([row["Plan"] for row in group]),
                "Time of day": str(hour),
                "Programmed Split (sec)": f"{median(float(row['Programmed Split (sec)']) for row in group):.2f}",
                "Average Split (sec)": f"{mean(float(row['Average Split (sec)']) for row in group):.2f}",
            }
        )

    return rebuilt_rows


def relabel_fulltrace(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    relabeled = []
    for row in rows:
        relabeled.append(
            {
                "Signal ID": row["Signal ID"],
                "Phase #": row["Phase #"],
                "Date": row["Date"],
                "Plan": row["Plan"],
                "Minute of day": row["Time of day"],
                "Programmed Split (sec)": row["Programmed Split (sec)"],
                "Average Split (sec)": row["Average Split (sec)"],
            }
        )
    return relabeled


def build_plan_periods(hourly_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_phase: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hourly_rows:
        by_phase[row["Phase #"]].append(row)

    periods: list[dict[str, str]] = []
    for phase in sorted(by_phase, key=int):
        rows = sorted(by_phase[phase], key=lambda row: int(row["Time of day"]))
        current: dict[str, object] | None = None

        for row in rows:
            hour = int(row["Time of day"])
            key = (row["Plan"], row["Programmed Split (sec)"])
            if current is None:
                current = {
                    "Signal ID": row["Signal ID"],
                    "Phase #": row["Phase #"],
                    "Date": row["Date"],
                    "Plan": row["Plan"],
                    "Start Hour": hour,
                    "End Hour": hour,
                    "Programmed Split (sec)": row["Programmed Split (sec)"],
                    "Average Split Values": [float(row["Average Split (sec)"])],
                }
                continue

            current_key = (current["Plan"], current["Programmed Split (sec)"])
            if key == current_key and hour == current["End Hour"] + 1:
                current["End Hour"] = hour
                current["Average Split Values"].append(float(row["Average Split (sec)"]))
                continue

            periods.append(
                {
                    "Signal ID": str(current["Signal ID"]),
                    "Phase #": str(current["Phase #"]),
                    "Date": str(current["Date"]),
                    "Plan": str(current["Plan"]),
                    "Start Hour": str(current["Start Hour"]),
                    "End Hour": str(current["End Hour"]),
                    "Programmed Split (sec)": str(current["Programmed Split (sec)"]),
                    "Average Split (sec)": f"{mean(current['Average Split Values']):.2f}",
                }
            )
            current = {
                "Signal ID": row["Signal ID"],
                "Phase #": row["Phase #"],
                "Date": row["Date"],
                "Plan": row["Plan"],
                "Start Hour": hour,
                "End Hour": hour,
                "Programmed Split (sec)": row["Programmed Split (sec)"],
                "Average Split Values": [float(row["Average Split (sec)"])],
            }

        if current is not None:
            periods.append(
                {
                    "Signal ID": str(current["Signal ID"]),
                    "Phase #": str(current["Phase #"]),
                    "Date": str(current["Date"]),
                    "Plan": str(current["Plan"]),
                    "Start Hour": str(current["Start Hour"]),
                    "End Hour": str(current["End Hour"]),
                    "Programmed Split (sec)": str(current["Programmed Split (sec)"]),
                    "Average Split (sec)": f"{mean(current['Average Split Values']):.2f}",
                }
            )

    return periods


def build_output_paths(fulltrace_csv: Path, output_dir: Path) -> dict[str, Path]:
    stem = fulltrace_csv.stem
    if stem.endswith("_fulltrace"):
        stem = stem[: -len("_fulltrace")]
    return {
        "hourly": output_dir / f"{stem}_hourly.csv",
        "hourly_corrected": output_dir / f"{stem}_hourly_corrected.csv",
        "hourly_original": output_dir / f"{stem}_hourly_original.csv",
        "minute_resolution": output_dir / f"{stem}_minute_resolution.csv",
        "plan_periods": output_dir / f"{stem}_plan_periods.csv",
    }


def compare_rows(
    original_rows: list[dict[str, str]],
    corrected_rows: list[dict[str, str]],
) -> int:
    by_key_original = {
        (row["Signal ID"], row["Phase #"], row["Date"], row["Time of day"]): row
        for row in original_rows
    }
    by_key_corrected = {
        (row["Signal ID"], row["Phase #"], row["Date"], row["Time of day"]): row
        for row in corrected_rows
    }

    changed = 0
    for key, corrected in by_key_corrected.items():
        original = by_key_original.get(key)
        if original is None:
            changed += 1
            continue
        if (
            original["Plan"] != corrected["Plan"]
            or original["Programmed Split (sec)"] != corrected["Programmed Split (sec)"]
            or original["Average Split (sec)"] != corrected["Average Split (sec)"]
        ):
            changed += 1
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild reliable Split Monitor hourly exports from minute-resolution fulltrace data."
    )
    parser.add_argument("fulltrace_csv", type=Path, help="Path to the source *_fulltrace.csv file.")
    parser.add_argument(
        "--hourly-csv",
        type=Path,
        help="Optional path to the existing hourly CSV. If provided, it will be archived and compared.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. Defaults to the fulltrace file directory.",
    )
    parser.add_argument(
        "--overwrite-hourly",
        action="store_true",
        help="Replace the main *_hourly.csv with the corrected hourly output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fulltrace_csv = args.fulltrace_csv.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else fulltrace_csv.parent
    paths = build_output_paths(fulltrace_csv, output_dir)

    fulltrace_rows = read_csv(fulltrace_csv)
    corrected_hourly_rows = rebuild_hourly_from_fulltrace(fulltrace_rows)
    minute_resolution_rows = relabel_fulltrace(fulltrace_rows)
    plan_period_rows = build_plan_periods(corrected_hourly_rows)

    changed_rows = None
    if args.hourly_csv:
        original_hourly_rows = read_csv(args.hourly_csv.resolve())
        write_csv(paths["hourly_original"], original_hourly_rows)
        changed_rows = compare_rows(original_hourly_rows, corrected_hourly_rows)

    write_csv(paths["hourly_corrected"], corrected_hourly_rows)
    write_csv(paths["minute_resolution"], minute_resolution_rows)
    write_csv(paths["plan_periods"], plan_period_rows)
    if args.overwrite_hourly:
        write_csv(paths["hourly"], corrected_hourly_rows)

    print(f"Fulltrace input: {fulltrace_csv}")
    print(f"Corrected hourly rows: {len(corrected_hourly_rows)}")
    print(f"Minute-resolution rows: {len(minute_resolution_rows)}")
    print(f"Plan-period rows: {len(plan_period_rows)}")
    print(f"Wrote corrected hourly: {paths['hourly_corrected']}")
    print(f"Wrote minute-resolution: {paths['minute_resolution']}")
    print(f"Wrote plan periods: {paths['plan_periods']}")
    if args.hourly_csv:
        print(f"Archived original hourly: {paths['hourly_original']}")
    if args.overwrite_hourly:
        print(f"Overwrote main hourly export: {paths['hourly']}")
    if changed_rows is not None:
        print(f"Rows changed versus original hourly: {changed_rows}")


if __name__ == "__main__":
    main()
