# ATM Intelligent Threat & Risk Analysis System

This project is a computer vision-based ATM security prototype that combines multiple detection modules to identify suspicious behavior and support risk-oriented surveillance.

## Overview

The system is designed as a conceptual ATM threat analysis solution. It uses multiple visual detection signals, including:

- face hiding detection
- hat detection
- risk-oriented detection logic

These signals can be combined to flag suspicious activity and support intelligent monitoring in ATM environments.

## Project Goals

- Detect suspicious visual behaviors around ATMs
- Combine multiple detection outputs into a risk-oriented analysis approach
- Explore how computer vision can support security systems in financial environments

## Detection Modules

### 1. Face Hiding Detection
Detects attempts to conceal identity in ATM environments.

### 2. Hat Detection
Detects headwear as an additional visual signal that may contribute to contextual risk analysis.

### 3. Risk Analysis Logic
Processes detection outputs and evaluates suspicious ATM scenarios using rule-based logic.

## Technologies Used

- Python
- YOLOv8
- Roboflow
- OpenCV

## Repository Structure

- `src/` → project source files, test scripts, and utility modules
- `configs/` → dataset YAML configuration files
- `docs/` → dataset export notes and Roboflow-related notes
- `datasets/` → information about omitted dataset files

## Dataset Note

Images, labels, videos, trained weights, and training splits are not included in this repository because of file size limitations.

## Status

This repository contains the project code, supporting utilities, and dataset configuration for an ATM intelligent threat and risk analysis prototype.

## Future Improvements

- integrate all detectors into a single inference pipeline
- calculate a clearer ATM risk score
- add real-time video analysis
- export threat alerts for suspicious events
