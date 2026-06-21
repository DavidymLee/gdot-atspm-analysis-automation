from __future__ import annotations

import asyncio
import csv
import re
import sys
from datetime import date
from pathlib import Path

import cv2
import pytesseract
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

URL = "https://traffic.dot.ga.gov/ATSPM/"
SIGNALS_TXT = ROOT / "signals.txt"
ANALYSIS_ROOT = ROOT / "analysis_plan_20251013" / "03_left_turn_gap"
IMAGES_DIR = ANALYSIS_ROOT / "images"
MANIFEST_CSV = ANALYSIS_ROOT / "left_turn_gap_manifest_20251013.csv"
RUN_LOG = ANALYSIS_ROOT / "run_log_20251013.txt"

RUN_DATE = date(2025, 10, 13)
DATE_TAG = RUN_DATE.strftime("%Y%m%d")
METRIC_VALUE = "31"
METRIC_LABEL = "Left Turn Gap Analysis"

MANIFEST_HEADER = [
    "signal_id",
    "date",
    "phase",
    "detector_type",
    "image_type",
    "chart_index",
    "width",
    "height",
    "relative_path",
    "status",
    "note",
]


def load_signal_ids() -> list[str]:
    signal_ids = []
    seen = set()
    for line in SIGNALS_TXT.read_text(encoding="utf-8").splitlines():
        match = re.search(r"\d+", line)
        if not match:
            continue
        signal_id = match.group(0)
        if signal_id not in seen:
            seen.add(signal_id)
            signal_ids.append(signal_id)
    return signal_ids


def mmddyyyy(d: date) -> str:
    return f"{d.month:02d}/{d.day:02d}/{d.year:04d}"


def sanitize_token(value: str) -> str:
    cleaned = value.strip().replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "", cleaned)
    return cleaned or "unknown"


def detector_token_from_text(detector_type: str) -> str:
    lowered = detector_type.lower().strip()
    if "lane-by-lane" in lowered or "lane by lane" in lowered:
        return "lane_by_lane_count"
    return sanitize_token(lowered.replace("-", " "))


def load_existing_rows() -> list[list[str]]:
    if not MANIFEST_CSV.exists():
        return []

    with MANIFEST_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            if row.get("date") != DATE_TAG:
                continue
            rows.append([row.get(column, "") for column in MANIFEST_HEADER])
        return rows


def signal_has_complete_existing_output(existing_rows: list[list[str]], signal_id: str) -> bool:
    success_rows = [
        row
        for row in existing_rows
        if row[0] == signal_id and row[1] == DATE_TAG and row[9] == "success"
    ]
    if not success_rows:
        return False

    for row in success_rows:
        relative_path = row[8].strip()
        if not relative_path:
            return False
        if not (IMAGES_DIR / relative_path).exists():
            return False
    return True


def extract_metadata_from_image(image_path: Path) -> tuple[str, str, str]:
    image = cv2.imread(str(image_path))
    if image is None:
        return "unknown", "unknown", "image_read_failed"

    crop = image[:220, :, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
    flattened = re.sub(r"\s+", " ", text).strip()

    phase = "unknown"
    for pattern in (
        r"Phase\s*#?\s*([0-9]+)",
        r"Phase\s*([0-9]+)",
        r"Phase([0-9]+)",
    ):
        match = re.search(pattern, flattened, re.IGNORECASE)
        if match:
            phase = match.group(1)
            break

    detector_type = "unknown"
    match = re.search(r"Detector Type:\s*(.+)", flattened, re.IGNORECASE)
    if match:
        detector_type = match.group(1).strip(" .;:")
    elif "Lane-By-Lane Count" in flattened:
        detector_type = "Lane-By-Lane Count"

    return phase, detector_type, flattened


async def process_signal(page, signal_id: str, rows: list[list[str]]) -> None:
    await page.goto(URL, wait_until="networkidle")
    await page.fill("#SignalID", signal_id)
    await page.click("#selectButton")

    try:
        await page.wait_for_selector("#MetricsList", timeout=20000)
    except Exception as exc:
        rows.append([signal_id, DATE_TAG, "", "", "", "", "", "", "", "failed", f"metrics_list_not_loaded: {exc}"])
        print(f"{signal_id}: could not load metrics list")
        return

    await page.select_option("#MetricsList", value=METRIC_VALUE)
    await page.wait_for_timeout(2000)
    await page.evaluate(
        f"""() => {{
            $('#StartDateDay').datepicker('setDate', '{mmddyyyy(RUN_DATE)}');
            $('#EndDateDay').datepicker('setDate', '{mmddyyyy(RUN_DATE)}');
            $('#StartTime').val('12:00');
            $('#StartAMPMddl').val('AM');
            $('#EndTime').val('11:59');
            $('#EndAMPMddl').val('PM');
            $('#ui-datepicker-div').hide();
        }}"""
    )
    await page.click("#CreateMetric")
    await page.wait_for_timeout(12000)

    images = await page.locator("#ReportPlaceHolder img").evaluate_all(
        """els => els.map((img, i) => ({
            index: i,
            src: img.src || '',
            width: img.naturalWidth || 0,
            height: img.naturalHeight || 0,
            alt: img.alt || ''
        }))"""
    )

    if not images:
        rows.append([signal_id, DATE_TAG, "", "", "", "", "", "", "", "failed", "no_report_images"])
        print(f"{signal_id}: no report images")
        return

    signal_dir = IMAGES_DIR / signal_id
    signal_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    saved_any = False

    for item in images:
        src = item["src"]
        if not src:
            continue

        response = await page.context.request.get(src)
        image_bytes = await response.body()

        tmp_path = signal_dir / f"_tmp_{signal_id}_{item['index']}_{DATE_TAG}.jpg"
        tmp_path.write_bytes(image_bytes)

        image_type = "legend" if item["height"] <= 150 else "chart"
        if image_type == "legend":
            phase = "legend"
            detector_type = ""
            filename = f"left_turn_gap_{signal_id}_legend_{DATE_TAG}.jpg"
            note = "Chart Legend"
        else:
            phase, detector_type, note = extract_metadata_from_image(tmp_path)
            detector_token = detector_token_from_text(detector_type)
            filename = f"left_turn_gap_{signal_id}_phase_{sanitize_token(phase)}_{detector_token}_{DATE_TAG}.jpg"
            if filename in used_names:
                filename = (
                    f"left_turn_gap_{signal_id}_phase_{sanitize_token(phase)}_"
                    f"{detector_token}_chart_{item['index']}_{DATE_TAG}.jpg"
                )

        used_names.add(filename)
        final_path = signal_dir / filename
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)
        relative_path = str(final_path.relative_to(IMAGES_DIR))

        rows.append(
            [
                signal_id,
                DATE_TAG,
                phase,
                detector_type,
                image_type,
                str(item["index"]),
                str(item["width"]),
                str(item["height"]),
                relative_path,
                "success",
                note[:300],
            ]
        )
        saved_any = True

    if saved_any:
        print(f"{signal_id}: saved left-turn-gap images")
    else:
        rows.append(
            [signal_id, DATE_TAG, "", "", "", "", "", "", "", "failed", "images_listed_but_not_saved"]
        )
        print(f"{signal_id}: images listed but nothing saved")


async def main() -> None:
    signal_ids = load_signal_ids()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    existing_rows = load_existing_rows()
    completed_signals = {
        signal_id for signal_id in signal_ids if signal_has_complete_existing_output(existing_rows, signal_id)
    }
    pending_signals = [signal_id for signal_id in signal_ids if signal_id not in completed_signals]
    pending_set = set(pending_signals)
    rows: list[list[str]] = [row for row in existing_rows if row[0] not in pending_set]

    print(f"Signals: {len(signal_ids)}")
    print(f"Metric: {METRIC_LABEL}")
    print(f"Date: {RUN_DATE.isoformat()}")
    print(f"Image output: {IMAGES_DIR}")
    print(f"Already complete: {len(completed_signals)}")
    print(f"Pending: {len(pending_signals)}")

    for signal_id in signal_ids:
        if signal_id in completed_signals:
            print(f"{signal_id}: already complete, skipping")

    if pending_signals:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(viewport={"width": 1920, "height": 1080})
            page = await context.new_page()
            page.set_default_navigation_timeout(60000)

            for signal_id in pending_signals:
                try:
                    await process_signal(page, signal_id, rows)
                except Exception as exc:
                    rows.append(
                        [signal_id, DATE_TAG, "", "", "", "", "", "", "", "failed", str(exc)]
                    )
                    print(f"{signal_id}: {exc}")

            await browser.close()
    else:
        print("Nothing pending for this signal list and date.")

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(rows)

    success_images = sum(1 for row in rows if row[9] == "success")
    failed_signals = len({row[0] for row in rows if row[9] == "failed"})
    RUN_LOG.write_text(
        "\n".join(
            [
                f"Signals attempted: {len(signal_ids)}",
                f"Metric: {METRIC_LABEL}",
                f"Date: {RUN_DATE.isoformat()}",
                f"Already complete before resume: {len(completed_signals)}",
                f"Pending before resume: {len(pending_signals)}",
                f"Success image rows: {success_images}",
                f"Failed signal count: {failed_signals}",
                f"Manifest: {MANIFEST_CSV}",
                f"Images dir: {IMAGES_DIR}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nManifest: {MANIFEST_CSV}")
    print(f"Run log: {RUN_LOG}")
    print(f"Already complete before resume: {len(completed_signals)}")
    print(f"Pending before resume: {len(pending_signals)}")
    print(f"Success image rows: {success_images}")
    print(f"Failed signal count: {failed_signals}")


if __name__ == "__main__":
    asyncio.run(main())
