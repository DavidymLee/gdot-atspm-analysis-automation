import asyncio
import csv
import os
import re
import sys
from datetime import date
from pathlib import Path

import cv2
import pytesseract
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

URL = 'https://traffic.dot.ga.gov/ATSPM/'
SIGNALS_TXT = ROOT / 'signals.txt'
ANALYSIS_ROOT = ROOT / 'analysis_plan_20251013' / '02_split_monitor'
IMAGES_DIR = ANALYSIS_ROOT / 'images'
CSV_DIR = ANALYSIS_ROOT / 'csv'
MANIFEST_CSV = ANALYSIS_ROOT / 'split_monitor_manifest_20251013.csv'
RUN_LOG = ANALYSIS_ROOT / 'run_log_20251013.txt'
RUN_DATE = date(2025, 10, 13)
DATE_STR = RUN_DATE.strftime('%m/%d/%Y')
DATE_TAG = RUN_DATE.strftime('%Y%m%d')
MANIFEST_HEADER = ['signal_id', 'date', 'phase', 'image_type', 'chart_index', 'filename', 'status', 'note']


def load_signal_ids() -> list[str]:
    ids = []
    seen = set()
    for line in SIGNALS_TXT.read_text(encoding='utf-8').splitlines():
        m = re.search(r'\d+', line)
        if not m:
            continue
        sid = m.group(0)
        if sid not in seen:
            seen.add(sid)
            ids.append(sid)
    return ids


def mmddyyyy(d: date) -> str:
    return f'{d.month:02d}/{d.day:02d}/{d.year:04d}'


def sanitize_token(value: str) -> str:
    value = value.strip().replace(' ', '_')
    value = re.sub(r'[^A-Za-z0-9_\-]+', '', value)
    return value or 'unknown'


def load_existing_rows() -> list[list[str]]:
    if not MANIFEST_CSV.exists():
        return []

    with MANIFEST_CSV.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if row.get('date') != DATE_TAG:
                continue
            rows.append([row.get(col, '') for col in MANIFEST_HEADER])
        return rows


def signal_has_complete_existing_output(existing_rows: list[list[str]], signal_id: str) -> bool:
    success_rows = [
        row for row in existing_rows
        if row[0] == signal_id and row[1] == DATE_TAG and row[6] == 'success'
    ]
    if not success_rows:
        return False

    for row in success_rows:
        filename = row[5].strip()
        if not filename:
            return False
        if not (IMAGES_DIR / filename).exists():
            return False
    return True


def extract_phase_from_image(image_path: Path) -> tuple[str, str]:
    img = cv2.imread(str(image_path))
    if img is None:
        return 'unknown', 'image_read_failed'

    crop = img[:180, :, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')

    m = re.search(r'Phase\s*#?\s*([0-9]+)', text, re.IGNORECASE)
    if not m:
        m = re.search(r'Phase\s*([0-9]+)', text, re.IGNORECASE)
    if not m:
        m = re.search(r'Phase([0-9]+)', text, re.IGNORECASE)

    if m:
        return m.group(1), text.strip()
    return 'unknown', text.strip()


async def get_metric_value(page, label: str):
    return await page.evaluate(
        """(label) => {
            let val = null;
            $('#MetricsList option').each(function() {
                if ($(this).text().indexOf(label) > -1) {
                    val = $(this).val();
                }
            });
            return val;
        }""",
        label,
    )


async def process_signal(page, signal_id: str, rows: list[list[str]]) -> None:
    await page.goto(URL, wait_until='networkidle')
    await page.fill('#SignalID', signal_id)
    await page.click('#selectButton')

    try:
        await page.wait_for_selector('#MetricsList', timeout=15000)
    except Exception as e:
        rows.append([signal_id, DATE_TAG, '', '', '', '', 'failed', f'metrics_list_not_loaded: {e}'])
        print(f'{signal_id}: could not load metrics list')
        return

    metric_val = await get_metric_value(page, 'Split Monitor')
    if not metric_val:
        rows.append([signal_id, DATE_TAG, '', '', '', '', 'failed', 'split_monitor_metric_not_found'])
        print(f'{signal_id}: Split Monitor metric not found')
        return

    await page.select_option('#MetricsList', value=metric_val)
    await page.evaluate(f"""() => {{
        $('#StartDateDay').datepicker('setDate', '{mmddyyyy(RUN_DATE)}');
        $('#EndDateDay').datepicker('setDate', '{mmddyyyy(RUN_DATE)}');
        $('#StartTime').val('12:00');
        $('#StartAMPMddl').val('AM');
        $('#EndTime').val('11:59');
        $('#EndAMPMddl').val('PM');
        $('#ui-datepicker-div').hide();
    }}""")
    await page.click('#CreateMetric')
    await page.wait_for_timeout(15000)

    imgs = await page.locator('#ReportPlaceHolder img').evaluate_all(
        """els => els.map((img, i) => {
            const r = img.getBoundingClientRect();
            return {
                index: i,
                src: img.src || '',
                width: Math.round(r.width),
                height: Math.round(r.height),
                naturalWidth: img.naturalWidth || 0,
                naturalHeight: img.naturalHeight || 0,
            };
        })"""
    )

    if not imgs:
        rows.append([signal_id, DATE_TAG, '', '', '', '', 'failed', 'no_report_images'])
        print(f'{signal_id}: no report images')
        return

    saved_any = False
    used_names: set[str] = set()

    for item in imgs:
        src = item['src']
        if not src:
            continue

        response = await page.context.request.get(src)
        img_bytes = await response.body()

        tmp_path = IMAGES_DIR / f'_tmp_{signal_id}_{item["index"]}_{DATE_TAG}.jpg'
        tmp_path.write_bytes(img_bytes)

        image_type = 'legend' if item['height'] <= 150 else 'chart'
        if image_type == 'legend':
            phase = 'legend'
            ocr_preview = 'Chart Legend'
            filename = f'split_monitor_{signal_id}_legend_{DATE_TAG}.jpg'
        else:
            phase, ocr_preview = extract_phase_from_image(tmp_path)
            filename = f'split_monitor_{signal_id}_phase_{sanitize_token(phase)}_{DATE_TAG}.jpg'
            if filename in used_names:
                filename = f'split_monitor_{signal_id}_phase_{sanitize_token(phase)}_chart_{item["index"]}_{DATE_TAG}.jpg'

        used_names.add(filename)
        final_path = IMAGES_DIR / filename
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)

        rows.append([
            signal_id,
            DATE_TAG,
            phase,
            image_type,
            item['index'],
            filename,
            'success',
            ocr_preview[:300].replace('\n', ' '),
        ])
        saved_any = True

    if saved_any:
        print(f'{signal_id}: saved split monitor images')
    else:
        rows.append([signal_id, DATE_TAG, '', '', '', '', 'failed', 'images_listed_but_not_saved'])
        print(f'{signal_id}: images listed but nothing saved')


async def main() -> None:
    signal_ids = load_signal_ids()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    existing_rows = load_existing_rows()
    completed_signals = {
        sid for sid in signal_ids if signal_has_complete_existing_output(existing_rows, sid)
    }
    pending_signals = [sid for sid in signal_ids if sid not in completed_signals]
    pending_signal_set = set(pending_signals)
    rows: list[list[str]] = [row for row in existing_rows if row[0] not in pending_signal_set]

    print(f'Signals: {len(signal_ids)}')
    print(f'Date: {RUN_DATE}')
    print(f'Image output: {IMAGES_DIR}')
    print(f'Already complete: {len(completed_signals)}')
    print(f'Pending: {len(pending_signals)}')

    for sid in signal_ids:
        if sid in completed_signals:
            print(f'{sid}: already complete, skipping')

    if pending_signals:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
            page = await context.new_page()
            page.set_default_navigation_timeout(60000)

            for sid in pending_signals:
                try:
                    await process_signal(page, sid, rows)
                except Exception as e:
                    rows.append([sid, DATE_TAG, '', '', '', '', 'failed', str(e)])
                    print(f'{sid}: {e}')

            await browser.close()
    else:
        print('Nothing pending for this signal list and date.')

    with MANIFEST_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(rows)

    success_images = sum(1 for r in rows if r[6] == 'success')
    failed_signals = len({r[0] for r in rows if r[6] == 'failed'})
    RUN_LOG.write_text(
        '\n'.join([
            f'Signals attempted: {len(signal_ids)}',
            f'Date: {RUN_DATE.isoformat()}',
            f'Already complete before resume: {len(completed_signals)}',
            f'Pending before resume: {len(pending_signals)}',
            f'Success image rows: {success_images}',
            f'Failed signal count: {failed_signals}',
            f'Manifest: {MANIFEST_CSV}',
            f'Images directory: {IMAGES_DIR}',
        ]) + '\n',
        encoding='utf-8',
    )

    print(f'\nManifest: {MANIFEST_CSV}')
    print(f'Run log: {RUN_LOG}')
    print(f'Already complete before resume: {len(completed_signals)}')
    print(f'Pending before resume: {len(pending_signals)}')
    print(f'Success image rows: {success_images}')
    print(f'Failed signal count: {failed_signals}')


if __name__ == '__main__':
    asyncio.run(main())
