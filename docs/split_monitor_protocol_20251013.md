# Split Monitor Protocol for October 13, 2025

## Scope
- Metric: `Split Monitor`
- Study area: `Downtown` signals from `signals.txt`
- Date: `10/13/2025`

## Raw Output Locations
- Full image set: `analysis_plan_20251013/02_split_monitor/images/`
- Full manifest: `analysis_plan_20251013/02_split_monitor/split_monitor_manifest_20251013.csv`
- Chart manifest: `analysis_plan_20251013/02_split_monitor/split_monitor_chart_manifest_20251013.csv`
- CSV template: `analysis_plan_20251013/02_split_monitor/csv/split_monitor_output_template_20251013.csv`

## Naming Structure
- Chart images are named as:
  - `split_monitor_{signal_id}_phase_{phase}_{yyyymmdd}.jpg`
- Example:
  - `split_monitor_7158_phase_2_20251013.jpg`
- If the local OCR phase hint was uncertain, the filename may contain `phase_unknown`.
  - Gemini should read the image title and correct the phase from the graph itself.

## Manifest Fields
- `signal_id`: downtown signal ID used in ATSPM
- `date`: collection date as `YYYYMMDD`
- `phase`: OCR phase hint from the chart title; may be `unknown`
- `chart_index`: image order within the report page
- `filename`: saved chart image filename
- `status`: success/failure of image capture
- `note`: OCR preview text used during naming

## Gemini Workflow
1. Upload one or more phase chart images from `images/`.
   Use files matching `split_monitor_*_phase_*.jpg` and skip `*_legend_*`.
2. Optionally provide the corresponding row(s) from `split_monitor_chart_manifest_20251013.csv` as metadata.
3. Use the Gemini prompt in `gemini_split_monitor_prompt.md`.
4. Ask Gemini to return CSV rows only.
5. Append Gemini output into the template schema:
   - `Signal ID`
   - `Phase #`
   - `Date`
   - `Plan`
   - `Time of day`
   - `Programmed Split (sec)`
   - `Average Split (sec)`

## Notes
- `images/` contains both phase charts and legend images; only use files with `_phase_` in the name for Gemini.
- `phase_unknown` filenames are still usable; Gemini should infer the correct phase from the chart title.
- If Gemini output is approximate, preserve the graph image and note the uncertainty in a separate QA field or review log.
