# Left-Turn Gap Analysis Protocol

## Objective

Download ATSPM left-turn gap analysis outputs for downtown signals for October 13, 2025 and convert the graph outputs into structured CSV files.

## Data Source

Georgia DOT ATSPM left-turn gap analysis charts for the downtown study signals.

## Folder Structure

- `analysis_plan_20251013/03_left_turn_gap/images`
- `analysis_plan_20251013/03_left_turn_gap/signal_exports`
- `analysis_plan_20251013/03_left_turn_gap/manifests`
- `analysis_plan_20251013/03_left_turn_gap/prompt`
- `analysis_plan_20251013/03_left_turn_gap/qa`

## Naming Convention

Each chart image should be saved using:

`left_turn_gap_<signalID>_phase_<phase>_<detectorType>_20251013.jpg`

This preserves:

- date
- signal ID
- phase
- detector type

## Required CSV Fields

- `Signal ID`
- `Phase #`
- `Detector Type`
- `Date`
- `Time (hour mark)`
- `Approximate Total Gaps Available`
- `1-3.3 (red) # of total gaps`
- `3.3-3.7 (lime green) # of total gaps`
- `3.7-7.4 (forest green) # of total gaps`
- `> 7.4 sec (turquoise) # of total gaps`
- `Percent of Green Time with safe gaps (blue line)`

## Current Status

The batch image download and CSV extraction have been completed for the available October 13, 2025 chart images.

The current practical step is QA review of the exported signal folders under `signal_exports/`, especially for signals where ATSPM did not return a usable chart image.
