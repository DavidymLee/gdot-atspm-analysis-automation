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
sys.path.insert(0, str(ROOT))

SIGNALS_TXT = ROOT / "signals.txt"
IMAGES_DIR = SCRIPT_DIR / "images"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "signal_exports"

PLOT_X1 = 126
PLOT_Y1 = 208
PLOT_X2 = 1286
PLOT_Y2 = 640
LEFT_AXIS_TOP_CROP = (72, 196, 118, 222)
HEADER_CROP = (0, 0, 1400, 205)

RIGHT_AXIS_MAX = 100.0
NOMINAL_BAR_WIDTH_PX = 10.0

SUMMARY_HEADER = [
    "signal_id",
    "phase",
    "detector_type",
    "date",
    "left_axis_max",
    "rows_written",
    "filename",
    "status",
    "note",
]


@dataclass(frozen=True)
class ChartImage:
    signal_id: str
    phase_token: str
    detector_token: str
    date_tag: str
    path: Path


@dataclass
class ChartSummary:
    signal_id: str
    phase: str
    detector_type: str
    date_text: str
    left_axis_max: float
    rows_written: int
    filename: str
    status: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ATSPM Left Turn Gap chart images into hourly CSV outputs."
    )
    parser.add_argument(
        "--signals-file",
        type=Path,
        default=SIGNALS_TXT,
        help="Path to the signal list. Defaults to signals.txt.",
    )
    parser.add_argument(
        "--signal-id",
        action="append",
        help="Optional signal ID filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=IMAGES_DIR,
        help="Directory containing left-turn-gap images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where extracted CSV outputs will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exports.",
    )
    return parser.parse_args()


def load_signal_ids(signals_file: Path, explicit_ids: list[str] | None) -> list[str]:
    if explicit_ids:
        signal_ids = []
        seen = set()
        for item in explicit_ids:
            match = re.search(r"\d+", item)
            if not match:
                continue
            signal_id = match.group(0)
            if signal_id not in seen:
                seen.add(signal_id)
                signal_ids.append(signal_id)
        return signal_ids

    signal_ids = []
    seen = set()
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
    by_signal: dict[str, list[ChartImage]] = defaultdict(list)
    signal_set = set(signal_ids)
    pattern = re.compile(r"^left_turn_gap_(\d+)_phase_([A-Za-z0-9]+)_(.+)_(\d{8})\.jpg$")

    for path in sorted(images_dir.rglob("left_turn_gap_*_phase_*_*.jpg")):
        match = pattern.match(path.name)
        if not match:
            continue
        signal_id, phase_token, detector_token, date_tag = match.groups()
        if signal_id not in signal_set:
            continue
        by_signal[signal_id].append(
            ChartImage(
                signal_id=signal_id,
                phase_token=phase_token,
                detector_token=detector_token,
                date_tag=date_tag,
                path=path,
            )
        )
    return by_signal


def normalize_detector_type(token: str) -> str:
    lowered = token.lower()
    if lowered == "lane_by_lane_count":
        return "Lane-By-Lane Count"
    if lowered == "unknown":
        return "unknown"
    words = lowered.replace("-", "_").split("_")
    return " ".join(word.capitalize() for word in words if word)


def detector_token_from_text(detector_type: str) -> str:
    lowered = detector_type.lower().strip()
    if "lane-by-lane" in lowered or "lane by lane" in lowered:
        return "lane_by_lane_count"
    cleaned = lowered.replace("-", " ").replace("/", " ")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
    return cleaned or "unknown"


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


def detect_phase_and_detector(image_bgr: np.ndarray, chart_image: ChartImage) -> tuple[str, str, str]:
    if chart_image.phase_token.isdigit() and chart_image.detector_token != "unknown":
        return (
            chart_image.phase_token,
            normalize_detector_type(chart_image.detector_token),
            "filename",
        )

    x1, y1, x2, y2 = HEADER_CROP
    crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    text = ocr_text(gray, config="--oem 3 --psm 6", timeout=3.0)

    phase = chart_image.phase_token
    for pattern in (
        r"Phase\s*#?\s*([0-9]+)",
        r"Phase\s*([0-9]+)",
        r"Phase([0-9]+)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            phase = match.group(1)
            break

    detector_type = normalize_detector_type(chart_image.detector_token)
    match = re.search(r"Detector Type:\s*(.+)", text, re.IGNORECASE)
    if match:
        detector_type = match.group(1).strip(" .;:")
    elif "Lane-By-Lane Count" in text:
        detector_type = "Lane-By-Lane Count"

    source = "ocr"
    if chart_image.phase_token.isdigit() and chart_image.detector_token != "unknown":
        source = "filename"
    return phase, detector_type, source


def detect_left_axis_max(image_bgr: np.ndarray) -> tuple[float, str]:
    x1, y1, x2, y2 = LEFT_AXIS_TOP_CROP
    crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    text = ocr_text(
        gray,
        config="--psm 7 -c tessedit_char_whitelist=0123456789",
        timeout=2.0,
    )
    digits = re.findall(r"\d+", text)
    if digits:
        value = float(digits[0])
        if value in {50.0, 100.0, 300.0}:
            return value, text or "ocr"
    return 300.0, text or "fallback_300"


def rgb_masks(plot_rgb: np.ndarray) -> dict[str, np.ndarray]:
    r = plot_rgb[:, :, 0]
    g = plot_rgb[:, :, 1]
    b = plot_rgb[:, :, 2]
    return {
        "red": (r > 160) & (g < 80) & (b < 80),
        "lime": (r > 90) & (g > 170) & (b < 100),
        "green": (r < 80) & (g > 70) & (g < 180) & (b < 90),
        "turquoise": (r < 100) & (g > 120) & (b > 120),
        "blue": build_blue_mask(plot_rgb),
    }


def is_blank_plot(plot_rgb: np.ndarray) -> bool:
    interior = plot_rgb[8:-8, 8:-8]
    if interior.size == 0:
        return True
    gray = cv2.cvtColor(interior, cv2.COLOR_RGB2GRAY)
    nonwhite_ratio = float(np.count_nonzero(gray < 245)) / float(gray.size)
    colorful_ratio = float(
        np.count_nonzero(interior.max(axis=2) - interior.min(axis=2) > 30)
    ) / float(gray.size)
    return nonwhite_ratio < 0.01 and colorful_ratio < 0.002


def build_blue_mask(plot_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    r = plot_rgb[:, :, 0].astype(np.int16)
    g = plot_rgb[:, :, 1].astype(np.int16)
    b = plot_rgb[:, :, 2].astype(np.int16)
    color_span = plot_rgb.max(axis=2).astype(np.int16) - plot_rgb.min(axis=2).astype(np.int16)

    hsv_mask = (h >= 100) & (h <= 140) & (s >= 60) & (v >= 30)
    dominant_mask = (b >= 40) & ((b - np.maximum(r, g)) >= 20) & (color_span >= 20)
    return hsv_mask | dominant_mask


def extract_blue_trace_values(plot_rgb: np.ndarray) -> tuple[np.ndarray, int]:
    blue_mask = build_blue_mask(plot_rgb)
    ys = np.full(plot_rgb.shape[1], np.nan, dtype=float)

    for x in range(plot_rgb.shape[1]):
        hits = np.where(blue_mask[:, x])[0]
        if hits.size:
            ys[x] = float(np.median(hits))

    x_positions = np.arange(len(ys))
    good = ~np.isnan(ys)
    if not good.any():
        raise ValueError("No blue trace pixels detected.")

    ys[~good] = np.interp(x_positions[~good], x_positions[good], ys[good])
    values = ((plot_rgb.shape[0] - 1 - ys) / (plot_rgb.shape[0] - 1)) * RIGHT_AXIS_MAX
    return values, int(good.sum())


def pixel_area_to_count(area_pixels: int, plot_height: int, left_axis_max: float) -> float:
    return (area_pixels / NOMINAL_BAR_WIDTH_PX) * (left_axis_max / (plot_height - 1))


def write_overlay(image_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    overlay = image.copy()

    cv2.rectangle(overlay, (PLOT_X1, PLOT_Y1), (PLOT_X2, PLOT_Y2), (0, 0, 255), 2)
    plot_width = PLOT_X2 - PLOT_X1 + 1
    for hour in range(25):
        x = PLOT_X1 + int(round((hour / 24.0) * plot_width))
        cv2.line(overlay, (x, PLOT_Y1), (x, PLOT_Y2), (180, 180, 180), 1)
        if hour < 24:
            cv2.putText(
                overlay,
                str(hour),
                (x + 3, PLOT_Y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def write_composite_mask(plot_rgb: np.ndarray, output_path: Path) -> None:
    masks = rgb_masks(plot_rgb)
    composite = np.zeros_like(plot_rgb)
    composite[masks["red"]] = (255, 0, 0)
    composite[masks["lime"]] = (170, 255, 0)
    composite[masks["green"]] = (0, 128, 0)
    composite[masks["turquoise"]] = (0, 200, 200)
    composite[masks["blue"]] = (0, 0, 255)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))


def build_rows_for_chart(path: Path, chart_image: ChartImage) -> tuple[list[dict[str, object]], ChartSummary, np.ndarray]:
    image_bgr = read_image(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    phase, detector_type, metadata_source = detect_phase_and_detector(image_bgr, chart_image)
    left_axis_max, axis_ocr = detect_left_axis_max(image_bgr)

    plot_rgb = image_rgb[PLOT_Y1 : PLOT_Y2 + 1, PLOT_X1 : PLOT_X2 + 1]
    masks = rgb_masks(plot_rgb)
    plot_height, plot_width = plot_rgb.shape[:2]
    blank_plot = is_blank_plot(plot_rgb)

    if blank_plot:
        blue_values = np.zeros(plot_width, dtype=float)
        blue_columns = 0
    else:
        blue_values, blue_columns = extract_blue_trace_values(plot_rgb)

    rows: list[dict[str, object]] = []
    for hour in range(24):
        x0 = int(round((hour / 24.0) * plot_width))
        x1 = int(round(((hour + 1) / 24.0) * plot_width))
        if x1 <= x0:
            x1 = x0 + 1

        if blank_plot:
            red_count = 0.0
            lime_count = 0.0
            green_count = 0.0
            turquoise_count = 0.0
            safe_gap_pct = 0.0
        else:
            red_count = pixel_area_to_count(int(masks["red"][:, x0:x1].sum()), plot_height, left_axis_max)
            lime_count = pixel_area_to_count(int(masks["lime"][:, x0:x1].sum()), plot_height, left_axis_max)
            green_count = pixel_area_to_count(int(masks["green"][:, x0:x1].sum()), plot_height, left_axis_max)
            turquoise_count = pixel_area_to_count(
                int(masks["turquoise"][:, x0:x1].sum()), plot_height, left_axis_max
            )
            safe_gap_pct = float(np.mean(blue_values[x0:x1]))

        total_count = red_count + lime_count + green_count + turquoise_count
        rows.append(
            {
                "Signal ID": chart_image.signal_id,
                "Phase #": phase,
                "Detector Type": detector_type,
                "Date": date_tag_to_iso(chart_image.date_tag),
                "Time (hour mark)": hour,
                "Approximate Total Gaps Available": round(total_count, 1),
                "1-3.3 (red) # of total gaps": round(red_count, 1),
                "3.3-3.7 (lime green) # of total gaps": round(lime_count, 1),
                "3.7-7.4 (forest green) # of total gaps": round(green_count, 1),
                "> 7.4 sec (turquoise) # of total gaps": round(turquoise_count, 1),
                "Percent of Green Time with safe gaps (blue line)": round(safe_gap_pct, 1),
            }
        )

    notes = [f"axis_ocr={axis_ocr or 'n/a'}"]
    if metadata_source != "filename":
        notes.append(f"metadata_source={metadata_source}")
    if blank_plot:
        notes.append("blank_plot_zero_fill")
    else:
        notes.append(f"blue_cols={blue_columns}")

    summary = ChartSummary(
        signal_id=chart_image.signal_id,
        phase=phase,
        detector_type=detector_type,
        date_text=date_tag_to_iso(chart_image.date_tag),
        left_axis_max=left_axis_max,
        rows_written=len(rows),
        filename=path.name,
        status="OK",
        note=";".join(notes),
    )
    return rows, summary, plot_rgb


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summaries: list[ChartSummary], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_HEADER)
        for summary in summaries:
            writer.writerow(
                [
                    summary.signal_id,
                    summary.phase,
                    summary.detector_type,
                    summary.date_text,
                    summary.left_axis_max,
                    summary.rows_written,
                    summary.filename,
                    summary.status,
                    summary.note,
                ]
            )


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def phase_sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):03d}") if value.isdigit() else (1, value)


def export_signal(
    signal_id: str,
    chart_images: list[ChartImage],
    output_dir: Path,
    overwrite: bool,
) -> dict[str, str]:
    signal_dir = output_dir / signal_id
    signal_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = signal_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = signal_dir / f"left_turn_gap_signal_{signal_id}_hourly.csv"
    summary_csv = signal_dir / f"left_turn_gap_signal_{signal_id}_summary.csv"

    if not overwrite and combined_csv.exists() and summary_csv.exists():
        existing_rows = read_summary(summary_csv)
        return {
            "signal_id": signal_id,
            "status": "skipped_existing",
            "charts_processed": str(len(existing_rows)),
            "phases": ",".join(row["phase"] for row in existing_rows),
            "detector_types": ",".join(sorted({row["detector_type"] for row in existing_rows})),
            "date": existing_rows[0]["date"] if existing_rows else "",
        }

    all_rows: list[dict[str, object]] = []
    summaries: list[ChartSummary] = []

    for chart_image in chart_images:
        rows, summary, plot_rgb = build_rows_for_chart(chart_image.path, chart_image)
        all_rows.extend(rows)
        summaries.append(summary)

        detector_token = detector_token_from_text(summary.detector_type)
        phase_csv = signal_dir / (
            f"left_turn_gap_{signal_id}_phase_{summary.phase}_{detector_token}_hourly.csv"
        )
        write_csv(rows, phase_csv)
        write_overlay(
            chart_image.path,
            qa_dir / f"left_turn_gap_{signal_id}_phase_{summary.phase}_{detector_token}_overlay.jpg",
        )
        write_composite_mask(
            plot_rgb,
            qa_dir / f"left_turn_gap_{signal_id}_phase_{summary.phase}_{detector_token}_mask.png",
        )

    all_rows.sort(
        key=lambda row: (
            phase_sort_key(str(row["Phase #"])),
            str(row["Detector Type"]),
            int(row["Time (hour mark)"]),
        )
    )

    write_csv(all_rows, combined_csv)
    write_summary(sorted(summaries, key=lambda s: (phase_sort_key(s.phase), s.detector_type)), summary_csv)

    return {
        "signal_id": signal_id,
        "status": "ok",
        "charts_processed": str(len(summaries)),
        "phases": ",".join(summary.phase for summary in sorted(summaries, key=lambda s: phase_sort_key(s.phase))),
        "detector_types": ",".join(sorted({summary.detector_type for summary in summaries})),
        "date": summaries[0].date_text if summaries else "",
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_map = collect_chart_images(args.images_dir.resolve(), signal_ids)
    batch_rows: list[dict[str, str]] = []
    completed = 0
    failed = 0

    for signal_id in signal_ids:
        chart_images = chart_map.get(signal_id, [])
        if not chart_images:
            batch_rows.append(
                {
                    "signal_id": signal_id,
                    "status": "missing_images",
                    "charts_processed": "0",
                    "phases": "",
                    "detector_types": "",
                    "date": "",
                }
            )
            failed += 1
            print(f"{signal_id}: no left-turn-gap chart images found")
            continue

        try:
            result = export_signal(signal_id, chart_images, output_dir, args.overwrite)
            batch_rows.append(result)
            completed += 1
            print(
                f"{signal_id}: {result['status']} "
                f"(charts={result['charts_processed']}, detectors={result['detector_types']})"
            )
        except Exception as exc:
            batch_rows.append(
                {
                    "signal_id": signal_id,
                    "status": f"failed: {exc}",
                    "charts_processed": "0",
                    "phases": "",
                    "detector_types": "",
                    "date": "",
                }
            )
            failed += 1
            print(f"{signal_id}: {exc}")

    write_batch_summary(output_dir / "left_turn_gap_batch_summary.csv", batch_rows)
    print(f"Signals requested: {len(signal_ids)}")
    print(f"Signals completed: {completed}")
    print(f"Signals failed/missing: {failed}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
