# GDOT ATSPM Analysis Automation

Python-based OCR and computer vision workflows for extracting GDOT ATSPM traffic signal performance metrics from chart-based visualizations and converting them into structured datasets for downstream traffic operations analysis.

## Overview

This repository showcases automation workflows developed for GDOT ATSPM analysis, with a focus on converting visual traffic signal performance reports into machine-readable outputs.

The project includes:
- automated download workflows for ATSPM chart-based reports
- OCR and computer vision pipelines for extracting values from visualizations
- structured CSV exports for traffic engineering analysis
- protocol and QA documentation used to support repeatable processing

## Project Highlights

- Extracted performance data from GDOT ATSPM visual reports using Python, OCR, and image-processing techniques
- Converted graphical traffic signal performance outputs into structured datasets
- Automated extraction of signal timing, split utilization, and gap availability metrics across multiple signal phases
- Built repeatable workflows for batch processing, validation, and export review

## Repository Structure

```text
gdot-atspm-analysis-automation/
├── split_monitor/
│   ├── extract_split_monitor_batch.py
│   ├── rebuild_split_monitor_exports.py
│   └── run_split_monitor_download_20251013.py
├── left_turn_gap/
│   ├── extract_left_turn_gap_batch.py
│   └── run_left_turn_gap_download_20251013.py
├── docs/
│   ├── split_monitor_protocol_20251013.md
│   └── left_turn_gap_protocol_20251013.md
├── example_outputs/
│   ├── split_monitor/
│   ├── left_turn_gap/
│   └── qa/
├── .gitignore
└── README.md
