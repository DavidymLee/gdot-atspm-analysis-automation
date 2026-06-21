from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from rebuild_split_monitor_exports import (  # noqa: E402
    build_plan_periods,
    rebuild_hourly_from_fulltrace,
    relabel_fulltrace,
    write_csv,
)

IMAGES_DIR = SCRIPT_DIR / "images"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "signal_exports"
SIGNALS_TXT = ROOT / "signals.txt"

PLOT_X1 = 126
PLOT_X2 = 1343
PLOT_Y1 = 255
PLOT_Y2 = 640
PLOT_WIDTH = PLOT_X2 - PLOT_X1 + 1
PLOT_HEIGHT = PLOT_Y2 - PLOT_Y1 + 1

PHASE_TITLE_CROP = (0, 0, 1400, 175)
PHASE_FOCUSED_CROP = (600, 120, 820, 170)
PLAN_ROW_Y1 = 176
PLAN_ROW_Y2 = 196
Y_AXIS_MAX_CROP = (55, 245, 125, 315)

HEADER_PLAN_THRESHOLD = 200
BACKGROUND_BLUE_THRESHOLD_MIN = 8.0
BACKGROUND_TINY_RUN_PX = 10
PROGRAM_LINE_BOTTOM_BUFFER_PX = 20
PROGRAM_LINE_BRIDGE_PX = 90
TRACE_NEIGHBORHOOD_PX = 3

SUMMARY_HEADER = [
    "signal_id",
    "phase",
    "date",
    "source_image",
    "status",
    "y_axis_max_seconds",
    "segment_count",
    "plan_sequence",
    "plan_sources",
    "notes",
]

FULLTRACE_HEADER = [
    "Signal ID",
    "Phase #",
    "Date",
    "Plan",
    "Time of day",
    "Programmed Split (sec)",
    "Average Split (sec)",
]


@dataclass(frozen=True)
class ChartImage:
    signal_id: str
    phase_token: str
    date_tag: str
    path: Path


@dataclass
class ChartExtract:
    image: ChartImage
    phase: str
    y_axis_max: float
    y_axis_note: str
    run_bounds: list[tuple[int, int]]
    per_segment_candidates: list[Counter[str]]
    programmed_trace: np.ndarray
    average_trace: np.ndarray
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Split Monitor chart images into fulltrace and corrected hourly CSV files."
    )
    parser.add_argument(
        "--signals-file",
        type=Path,
        default=SIGNALS_TXT,
        help="Path to the signal list. Defaults to the repo's signals.txt.",
    )
    parser.add_argument(
        "--signal-id",
        action="append",
        help="Optional signal ID filter. Can be provided multiple times.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=IMAGES_DIR,
        help="Directory containing Split Monitor chart images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-signal exports will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-signal exports.",
    )
    return parser.parse_args()


def load_signal_ids(signals_file: Path, explicit_ids: list[str] | None) -> list[str]:
    if explicit_ids:
        seen: set[str] = set()
        ids: list[str] = []
        for item in explicit_ids:
            match = re.search(r"\d+", item)
            if not match:
                continue
            signal_id = match.group(0)
            if signal_id not in seen:
                seen.add(signal_id)
                ids.append(signal_id)
        return ids

    seen = set()
    signal_ids = []
    for line in signals_file.read_text(encoding="utf-8").splitlines():
        match = re.search(r"\d+", line)
        if not match:
            continue
        signal_id = match.group(0)
        if signal_id not in seen:
            seen.add(signal_id)
            signal_ids.append(signal_id)
    return signal_ids


def collect_chart_images(images_dir: Path, signal_ids: list[str]) -> dict[str, list[ChartImage]]:
    signal_set = set(signal_ids)
    by_signal: dict[str, list[ChartImage]] = defaultdict(list)
    pattern = re.compile(r"^split_monitor_(\d+)_phase_([A-Za-z0-9]+)_(\d{8})\.jpg$")

    for path in sorted(images_dir.glob("split_monitor_*_phase_*_*.jpg")):
        match = pattern.match(path.name)
        if not match:
            continue
        signal_id, phase_token, date_tag = match.groups()
        if signal_id not in signal_set:
            continue
        if phase_token == "legend":
            continue
        by_signal[signal_id].append(
            ChartImage(
                signal_id=signal_id,
                phase_token=phase_token,
                date_tag=date_tag,
                path=path,
            )
        )
    return by_signal


def date_tag_to_iso(date_tag: str) -> str:
    return f"{date_tag[0:4]}-{date_tag[4:6]}-{date_tag[6:8]}"


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def ocr_text(image_gray: np.ndarray, *, config: str, timeout: float = 5.0) -> str:
    text = pytesseract.image_to_string(image_gray, config=config, timeout=timeout)
    return re.sub(r"\s+", " ", text).strip()


def detect_phase(image_bgr: np.ndarray, phase_token: str) -> tuple[str, str]:
    if phase_token.isdigit():
        return phase_token, "filename"

    x1, y1, x2, y2 = PHASE_TITLE_CROP
    crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    text = ocr_text(gray, config="--oem 3 --psm 6", timeout=3.0)

    for source_name, candidate_text in (("ocr", text),):
        for pattern in (
            r"Phase\s*#?\s*([0-9]+)",
            r"Phase\s*([0-9]+)",
            r"Phase([0-9]+)",
        ):
            match = re.search(pattern, candidate_text, re.IGNORECASE)
            if match:
                return match.group(1), source_name

    x1, y1, x2, y2 = PHASE_FOCUSED_CROP
    focused_crop = image_bgr[y1:y2, x1:x2]
    focused_gray = cv2.cvtColor(focused_crop, cv2.COLOR_BGR2GRAY)
    focused_gray = cv2.resize(focused_gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    focused_text = ocr_text(focused_gray, config="--oem 3 --psm 7", timeout=2.0)
    for pattern in (
        r"Phase\s*#?\s*([0-9]+)",
        r"Phase\s*([0-9]+)",
        r"Phase([0-9]+)",
    ):
        match = re.search(pattern, focused_text, re.IGNORECASE)
        if match:
            return match.group(1), "ocr_focused"
    return phase_token, "fallback_phase_token"


def phase_sort_key(phase: str) -> tuple[int, str]:
    return (0, f"{int(phase):03d}") if phase.isdigit() else (1, phase)


def detect_y_axis_max(image_bgr: np.ndarray) -> tuple[float, str]:
    x1, y1, x2, y2 = Y_AXIS_MAX_CROP
    crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)

    candidates: list[int] = []
    notes: list[str] = []
    for threshold in (180, 200):
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        text = ocr_text(
            binary,
            config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789",
            timeout=2.0,
        )
        notes.append(f"th{threshold}={text or 'blank'}")
        for token in re.findall(r"\d+", text):
            value = int(token)
            if 20 <= value <= 300:
                candidates.append(value)

    if candidates:
        value = max(candidates)
        return float(value), ";".join(notes)

    return 100.0, ";".join(notes + ["fallback=100"])


def detect_background_runs(image_bgr: np.ndarray) -> list[tuple[int, int]]:
    band = image_bgr[PLOT_Y1 + 5 : PLOT_Y1 + 45, PLOT_X1 : PLOT_X2 + 1]
    column_metric = np.zeros(PLOT_WIDTH, dtype=float)

    for x in range(PLOT_WIDTH):
        column = band[:, x, :].reshape(-1, 3)
        bright = column[column.mean(axis=1) > 220]
        if len(bright) == 0:
            bright = column
        median_bgr = np.median(bright, axis=0)
        column_metric[x] = float(median_bgr[0] - median_bgr[2])

    smooth = np.convolve(column_metric, np.ones(9) / 9, mode="same")
    threshold = max(
        BACKGROUND_BLUE_THRESHOLD_MIN,
        float(np.percentile(smooth, 25) + np.percentile(smooth, 75)) / 2.0,
    )
    blue_mask = smooth > threshold

    runs: list[tuple[int, int, bool]] = []
    run_start = 0
    current_value = bool(blue_mask[0])
    for x in range(1, PLOT_WIDTH):
        value = bool(blue_mask[x])
        if value != current_value:
            runs.append((run_start, x - 1, current_value))
            run_start = x
            current_value = value
    runs.append((run_start, PLOT_WIDTH - 1, current_value))

    merged = list(runs)
    while True:
        changed = False
        next_runs: list[tuple[int, int, bool]] = []
        i = 0
        while i < len(merged):
            start, end, is_blue = merged[i]
            width = end - start + 1
            if width < BACKGROUND_TINY_RUN_PX and len(merged) > 1:
                changed = True
                if i == 0:
                    n_start, n_end, n_color = merged[i + 1]
                    merged[i + 1] = (start, n_end, n_color)
                elif i == len(merged) - 1:
                    p_start, p_end, p_color = next_runs[-1]
                    next_runs[-1] = (p_start, end, p_color)
                else:
                    prev_width = next_runs[-1][1] - next_runs[-1][0] + 1
                    next_width = merged[i + 1][1] - merged[i + 1][0] + 1
                    if prev_width >= next_width:
                        p_start, _, p_color = next_runs[-1]
                        next_runs[-1] = (p_start, end, p_color)
                    else:
                        n_start, n_end, n_color = merged[i + 1]
                        merged[i + 1] = (start, n_end, n_color)
                i += 1
                continue

            next_runs.append((start, end, is_blue))
            i += 1

        merged = next_runs
        if not changed:
            break

    return [(start, end) for start, end, _ in merged]


def extract_numeric_candidates_from_text(text: str) -> Counter[str]:
    candidates: Counter[str] = Counter()
    if not text:
        return candidates

    for token in re.findall(r"\d{1,2}", text):
        normalized = str(int(token))
        candidates[normalized] += 2

    compact_digits = "".join(ch for ch in text if ch.isdigit())
    if 1 <= len(compact_digits) <= 2:
        candidates[str(int(compact_digits))] += 3

    for left, right in re.findall(r"(\d)\D{0,2}(\d)", text):
        candidates[str(int(left + right))] += 2

    return candidates


def scan_plan_candidates_for_run(
    image_bgr: np.ndarray,
    run_bounds: tuple[int, int],
) -> Counter[str]:
    start_rel, end_rel = run_bounds
    start_x = PLOT_X1 + start_rel
    end_x = PLOT_X1 + end_rel
    center_x = (start_x + end_x) // 2

    candidates: Counter[str] = Counter()
    for window_width in (50, 70, 110):
        left = max(0, center_x - window_width // 2)
        right = min(image_bgr.shape[1], center_x + window_width // 2)
        crop = image_bgr[PLAN_ROW_Y1:PLAN_ROW_Y2, left:right]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

        _, binary = cv2.threshold(gray, HEADER_PLAN_THRESHOLD, 255, cv2.THRESH_BINARY)
        text = ocr_text(binary, config="--oem 3 --psm 7", timeout=2.0)
        candidates.update(extract_numeric_candidates_from_text(text))

    return candidates


def choose_consensus_runs(charts: list[ChartExtract]) -> tuple[list[tuple[int, int]], str]:
    counts = Counter(len(chart.run_bounds) for chart in charts)
    target_count = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    matching = [chart for chart in charts if len(chart.run_bounds) == target_count]

    consensus: list[tuple[int, int]] = []
    for idx in range(target_count):
        starts = [chart.run_bounds[idx][0] for chart in matching]
        ends = [chart.run_bounds[idx][1] for chart in matching]
        consensus.append((int(round(np.median(starts))), int(round(np.median(ends)))))

    note = f"mode_segment_count={target_count};matching_charts={len(matching)}/{len(charts)}"
    return consensus, note


def resolve_plan_sequence(
    per_segment_counters: list[Counter[str]],
) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    source_notes: list[str] = []

    for idx, counter in enumerate(per_segment_counters):
        if not counter:
            resolved.append(str(idx + 1))
            source_notes.append("segment_fallback")
            continue

        ranked = sorted(counter.items(), key=lambda item: (-item[1], int(item[0])))
        chosen = ranked[0][0]

        if resolved and chosen == resolved[-1]:
            for candidate, weight in ranked[1:]:
                if candidate != resolved[-1] and weight >= max(1, ranked[0][1] // 5):
                    chosen = candidate
                    break

        if (
            len(chosen) == 1
            and any(len(candidate) == 2 and candidate.endswith(chosen) for candidate, _ in ranked)
        ):
            for candidate, weight in ranked:
                if len(candidate) == 2 and candidate.endswith(chosen):
                    chosen = candidate
                    break

        resolved.append(chosen)
        top_preview = ",".join(f"{candidate}:{weight}" for candidate, weight in ranked[:3])
        source_notes.append(f"ocr_vote[{top_preview}]")

    if "47" in resolved:
        for idx, counter in enumerate(per_segment_counters):
            chosen = resolved[idx]
            if chosen == "47" or not chosen.endswith("7"):
                continue
            support_47 = counter.get("47", 0)
            support_chosen = counter.get(chosen, 0)
            if support_47 and support_47 >= max(10, int(round(support_chosen * 0.5))):
                resolved[idx] = "47"
                source_notes[idx] += ";normalized_to_47"

    return resolved, source_notes


def build_programmed_trace(plot_bgr: np.ndarray, y_axis_max: float) -> np.ndarray:
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    warm_mask = ((h >= 5) & (h <= 35) & (s >= 50) & (v >= 80)).astype("uint8")

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(warm_mask, 8)
    filtered = np.zeros_like(warm_mask)
    for idx in range(1, component_count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        width = int(stats[idx, cv2.CC_STAT_WIDTH])
        height = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area >= 8 or width >= 8 or height >= 8:
            filtered[labels == idx] = 1

    upper_values = np.full(PLOT_WIDTH, np.nan)
    bottom_values = np.full(PLOT_WIDTH, np.nan)
    for x in range(PLOT_WIDTH):
        ys = np.where(filtered[:, x])[0]
        if len(ys) == 0:
            continue

        upper = ys[ys < PLOT_HEIGHT - PROGRAM_LINE_BOTTOM_BUFFER_PX]
        bottom = ys[ys >= PLOT_HEIGHT - PROGRAM_LINE_BOTTOM_BUFFER_PX]

        if len(upper):
            bins = np.bincount(upper // 4)
            peak_bin = int(bins.argmax())
            cluster = upper[(upper // 4) == peak_bin]
            upper_values[x] = float(np.median(cluster))

        if len(bottom):
            bottom_values[x] = float(np.median(bottom))

    traced = np.full(PLOT_WIDTH, np.nan)
    previous_y = np.nan
    last_upper_x = -10_000
    for x in range(PLOT_WIDTH):
        if not np.isnan(upper_values[x]):
            traced[x] = upper_values[x]
            previous_y = traced[x]
            last_upper_x = x
        elif (
            not np.isnan(previous_y)
            and previous_y < PLOT_HEIGHT - PROGRAM_LINE_BOTTOM_BUFFER_PX
            and (x - last_upper_x) <= PROGRAM_LINE_BRIDGE_PX
        ):
            traced[x] = previous_y
        elif not np.isnan(bottom_values[x]):
            traced[x] = bottom_values[x]
            previous_y = traced[x]
        elif not np.isnan(previous_y):
            traced[x] = previous_y

    if np.isnan(traced).all():
        raise ValueError("Programmed split line could not be detected.")

    x_positions = np.arange(PLOT_WIDTH)
    good = ~np.isnan(traced)
    traced[~good] = np.interp(x_positions[~good], x_positions[good], traced[good])
    return ((PLOT_HEIGHT - 1 - traced) / (PLOT_HEIGHT - 1)) * y_axis_max


def build_average_trace(plot_bgr: np.ndarray, y_axis_max: float) -> np.ndarray:
    plot_rgb = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = plot_rgb[:, :, 0], plot_rgb[:, :, 1], plot_rgb[:, :, 2]

    masks = {
        "blue": (r < 100) & (g < 100) & (b > 120),
        "green": (g > 80) & (r < 180) & (b < 180) & (g > r + 10) & (g > b + 10),
        "red": (r > 150) & (g < 100) & (b < 100),
    }

    points: list[tuple[float, float]] = []
    for mask in masks.values():
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
        for idx in range(1, component_count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if not (1 <= area <= 50):
                continue
            x, y = centroids[idx]
            if y >= PLOT_HEIGHT - 8:
                continue
            points.append((float(x), float(y)))

    if not points:
        raise ValueError("Termination event points could not be detected.")

    values = np.full(PLOT_WIDTH, np.nan)
    for x in range(PLOT_WIDTH):
        nearby = [y for px, y in points if abs(px - x) <= TRACE_NEIGHBORHOOD_PX]
        if nearby:
            values[x] = float(np.mean(nearby))

    x_positions = np.arange(PLOT_WIDTH)
    good = ~np.isnan(values)
    values[~good] = np.interp(x_positions[~good], x_positions[good], values[good])
    return ((PLOT_HEIGHT - 1 - values) / (PLOT_HEIGHT - 1)) * y_axis_max


def is_blank_plot(plot_bgr: np.ndarray) -> bool:
    interior = plot_bgr[8:-8, 8:-8]
    if interior.size == 0:
        return True

    gray = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)
    nonwhite_ratio = float(np.count_nonzero(gray < 245)) / float(gray.size)
    colorful_ratio = float(
        np.count_nonzero(interior.max(axis=2) - interior.min(axis=2) > 30)
    ) / float(gray.size)
    return nonwhite_ratio < 0.01 and colorful_ratio < 0.002


def extract_chart(image: ChartImage) -> ChartExtract:
    image_bgr = read_image(image.path)
    phase, phase_source = detect_phase(image_bgr, image.phase_token)
    y_axis_max, y_axis_note = detect_y_axis_max(image_bgr)
    run_bounds = detect_background_runs(image_bgr)
    plot_bgr = image_bgr[PLOT_Y1 : PLOT_Y2 + 1, PLOT_X1 : PLOT_X2 + 1]
    blank_plot = is_blank_plot(plot_bgr)

    notes: list[str] = []
    if phase_source != "filename":
        notes.append(f"phase_source={phase_source}")
    if "fallback" in y_axis_note:
        notes.append("y_axis_max_fallback")

    if blank_plot:
        programmed_trace = np.zeros(PLOT_WIDTH, dtype=float)
        average_trace = np.zeros(PLOT_WIDTH, dtype=float)
        notes.append("blank_plot_zero_fill")
    else:
        programmed_trace = build_programmed_trace(plot_bgr, y_axis_max)
        average_trace = build_average_trace(plot_bgr, y_axis_max)

    per_segment_candidates = [scan_plan_candidates_for_run(image_bgr, run) for run in run_bounds]

    return ChartExtract(
        image=image,
        phase=phase,
        y_axis_max=y_axis_max,
        y_axis_note=y_axis_note,
        run_bounds=run_bounds,
        per_segment_candidates=per_segment_candidates,
        programmed_trace=programmed_trace,
        average_trace=average_trace,
        notes=notes,
    )


def minute_for_column(column_index: int) -> int:
    return int(round(column_index * 1440.0 / (PLOT_WIDTH - 1)))


def plan_for_column(column_index: int, plan_runs: list[tuple[int, int]], plans: list[str]) -> str:
    for idx, (start, end) in enumerate(plan_runs):
        if start <= column_index <= end:
            return plans[idx]
    return plans[-1]


def build_fulltrace_rows(
    signal_id: str,
    date_text: str,
    charts: list[ChartExtract],
    plan_runs: list[tuple[int, int]],
    plans: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    charts_sorted = sorted(charts, key=lambda chart: phase_sort_key(chart.phase))

    for chart in charts_sorted:
        for x in range(PLOT_WIDTH):
            rows.append(
                {
                    "Signal ID": signal_id,
                    "Phase #": chart.phase,
                    "Date": date_text,
                    "Plan": plan_for_column(x, plan_runs, plans),
                    "Time of day": str(minute_for_column(x)),
                    "Programmed Split (sec)": f"{chart.programmed_trace[x]:.2f}",
                    "Average Split (sec)": f"{chart.average_trace[x]:.2f}",
                }
            )
    return rows


def build_summary_rows(
    charts: list[ChartExtract],
    date_text: str,
    plans: list[str],
    plan_source_notes: list[str],
    plan_mode_note: str,
    plan_runs: list[tuple[int, int]],
) -> list[dict[str, str]]:
    plan_sequence = ";".join(plans)
    segment_count = str(len(plan_runs))
    source_joined = ";".join(plan_source_notes)

    rows = []
    for chart in sorted(charts, key=lambda item: phase_sort_key(item.phase)):
        row_notes = list(chart.notes)
        if len(chart.run_bounds) != len(plan_runs):
            row_notes.append(
                f"chart_segments={len(chart.run_bounds)};consensus_segments={len(plan_runs)}"
            )
        row_notes.append(plan_mode_note)
        row_notes.append(chart.y_axis_note)
        rows.append(
            {
                "signal_id": chart.image.signal_id,
                "phase": chart.phase,
                "date": date_text,
                "source_image": chart.image.path.name,
                "status": "OK",
                "y_axis_max_seconds": f"{chart.y_axis_max:.2f}",
                "segment_count": segment_count,
                "plan_sequence": plan_sequence,
                "plan_sources": source_joined,
                "notes": ";".join(note for note in row_notes if note),
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def export_signal(
    signal_id: str,
    chart_images: list[ChartImage],
    output_dir: Path,
    overwrite: bool,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    signal_dir = output_dir / signal_id
    signal_dir.mkdir(parents=True, exist_ok=True)

    date_tags = Counter(chart.date_tag for chart in chart_images)
    date_tag = date_tags.most_common(1)[0][0]
    date_text = date_tag_to_iso(date_tag)

    fulltrace_path = signal_dir / f"split_monitor_signal_{signal_id}_fulltrace.csv"
    hourly_path = signal_dir / f"split_monitor_signal_{signal_id}_hourly.csv"
    minute_path = signal_dir / f"split_monitor_signal_{signal_id}_minute_resolution.csv"
    plan_period_path = signal_dir / f"split_monitor_signal_{signal_id}_plan_periods.csv"
    summary_path = signal_dir / f"split_monitor_signal_{signal_id}_summary.csv"

    output_paths = [fulltrace_path, hourly_path, minute_path, plan_period_path, summary_path]
    if not overwrite and all(path.exists() for path in output_paths):
        existing_rows = read_summary_csv(summary_path)
        return [], {
            "signal_id": signal_id,
            "status": "skipped_existing",
            "charts_processed": str(len(existing_rows)),
            "phases": ",".join(row["phase"] for row in existing_rows),
            "date": existing_rows[0]["date"] if existing_rows else date_text,
            "segments": existing_rows[0]["segment_count"] if existing_rows else "",
        }

    charts = [extract_chart(chart_image) for chart_image in chart_images if chart_image.date_tag == date_tag]
    if not charts:
        raise ValueError(f"No charts available for signal {signal_id} on date {date_tag}.")

    plan_runs, plan_mode_note = choose_consensus_runs(charts)
    per_segment_counters: list[Counter[str]] = [Counter() for _ in range(len(plan_runs))]
    for chart in charts:
        if len(chart.per_segment_candidates) != len(plan_runs):
            continue
        for idx, counter in enumerate(chart.per_segment_candidates):
            per_segment_counters[idx].update(counter)

    plans, plan_source_notes = resolve_plan_sequence(per_segment_counters)

    fulltrace_rows = build_fulltrace_rows(signal_id, date_text, charts, plan_runs, plans)
    hourly_rows = rebuild_hourly_from_fulltrace(fulltrace_rows)
    minute_rows = relabel_fulltrace(fulltrace_rows)
    plan_period_rows = build_plan_periods(hourly_rows)
    summary_rows = build_summary_rows(
        charts,
        date_text,
        plans,
        plan_source_notes,
        plan_mode_note,
        plan_runs,
    )

    write_csv(fulltrace_path, fulltrace_rows)
    write_csv(hourly_path, hourly_rows)
    write_csv(minute_path, minute_rows)
    write_csv(plan_period_path, plan_period_rows)
    write_summary_csv(summary_path, summary_rows)

    return summary_rows, {
        "signal_id": signal_id,
        "status": "ok",
        "charts_processed": str(len(charts)),
        "phases": ",".join(chart.phase for chart in sorted(charts, key=lambda item: phase_sort_key(item.phase))),
        "date": date_text,
        "segments": str(len(plan_runs)),
    }


def write_batch_summary(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    signal_ids = load_signal_ids(args.signals_file.resolve(), args.signal_id)
    image_map = collect_chart_images(args.images_dir.resolve(), signal_ids)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_rows: list[dict[str, str]] = []
    processed = 0
    failed = 0

    for signal_id in signal_ids:
        chart_images = image_map.get(signal_id, [])
        if not chart_images:
            failed += 1
            batch_rows.append(
                {
                    "signal_id": signal_id,
                    "status": "missing_images",
                    "charts_processed": "0",
                    "phases": "",
                    "date": "",
                    "segments": "",
                }
            )
            print(f"{signal_id}: no Split Monitor chart images found")
            continue

        try:
            _, batch_row = export_signal(signal_id, chart_images, output_dir, args.overwrite)
            batch_rows.append(batch_row)
            processed += 1
            print(
                f"{signal_id}: {batch_row['status']} "
                f"(charts={batch_row['charts_processed']}, segments={batch_row['segments']})"
            )
        except Exception as exc:
            failed += 1
            batch_rows.append(
                {
                    "signal_id": signal_id,
                    "status": f"failed: {exc}",
                    "charts_processed": "0",
                    "phases": "",
                    "date": "",
                    "segments": "",
                }
            )
            print(f"{signal_id}: {exc}")

    write_batch_summary(output_dir / "split_monitor_batch_summary.csv", batch_rows)
    print(f"Signals requested: {len(signal_ids)}")
    print(f"Signals completed: {processed}")
    print(f"Signals failed/missing: {failed}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
