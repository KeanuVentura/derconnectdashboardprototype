# DERConnect Evidence Triage — synthetic capstone prototype

**This project uses 100% fictional demand and meter IDs. It is not connected to DERConnect and does not diagnose actual campus equipment.**

## Decision supported
A UCSD facilities energy manager needs to decide which unusual submeter readings deserve an operator investigation. The synchronized interface connects a prioritized event queue, an actual-vs-expected load comparison, context checks and estimated positive-residual energy impact.

## Running locally

```bash
python -m pip install -r requirements.txt
python scripts/pipeline.py
python -m pytest -q tests
npm install
npm run dev
```

The Vite console prints a local URL. For GitHub Pages, set Pages to deploy built content or configure Actions to run `npm ci && npm run build` and publish `dist/`; Vite uses relative asset URLs. **The frontend could not be built in the preparation environment because npm installation timed out.** The source has not been browser-tested.

## Simulation
- Seed: `67`; 3 **fictional** buildings, 6 submeters, 15-minute intervals; 70 days total.
- 56 prior days for fixed historical comparison; 14 days for held-out anomaly detection.
- HVAC, lighting and plug profiles combine weekday/hour occupancy, weather-sensitive demand, meter-specific stochastic variance and schedule overrides.
- Four injected anomalies: sustained offset, ramp, after-hours offset and single-hour load pulse. The injected causes are descriptions of simulation mechanics, **not diagnosed physical causes**.
- One legitimate occupancy/schedule change has a distinct schedule flag and adjusted expectation.
- Missing readings remain missing; missing data are not counted as zero power.
- `public/data/ground_truth.json` remains separate from public-facing alert detection fixture. The UI does not use ground truth for triage.

## Method
Baseline for a given meter and 15-minute timeslot: historical median for matching weekday, hour and occupancy, using 56 pre-test days. At least 3 matching analogs required; fallback to a meter-and-occupancy median. A *fixed synthetic weather coefficient* of 0.3 kW/°C above 23°C is applied after removing it from the training analogs. This coefficient is illustrative, not estimated from DERConnect. Logged schedule overrides are corrected with a known synthetic multiplier.

Positive residual flags require demand exceeding `max(3.5 × robust train MAD scale, 0.22 × expected kW)`; detected episodes require at least two consecutive samples. Positive residual energy sums `max(0, measured_kw − expected_kw) × 0.25 hours` **only over flagged episode intervals**. This is not necessarily the full energy impact of a physical event. It is a model-dependent estimate, not financial loss or a measure of achievable savings.

Priority = `min(100, 8 log(1 + flagged excess kWh) + 4 log(1 + duration hours) + 2 log(1 + peak deviation kW))`. This is a deliberately uncalibrated heuristic and does not claim to measure safety severity, failure probability, or real-world operating risk.

## Schema
`observations.csv`: `timestamp` (UTC interval start), `building_id`, `building_name`, `meter_id`, `category`, `power_kw` (measured average over interval), `expected_kw` (model estimate), `temperature_c`, `occupied` (simulated state), `scheduled_change` (simulated operation), `quality`, `interval_hours`, `residual_kw`, `positive_excess_kwh`, `history_n`. Building consumption kW can be summed from *non-overlapping* submeters; a real hierarchy must be verified before aggregation.

`alerts.json`: `event_id`, meter/building, start/end (half-open UTC boundaries), duration, estimated excess energy, peak residual, mean deviation, ranking score and quality. `ground_truth.json` holds injected-event labels and simulation settings separately from detection outputs. `evaluation.json` contains executed holdout benchmark results.

## Executed checks
`pytest`: 6 passed on the authorship environment. Deterministic series generation, 15-minute interval and nonnegative power, energy integration, submeter-to-building sum, no training-period alerts, schedule changes, missing observations and holdout events are exercised.

Synthetic holdout at fixed seed: **4 true-positive events, 1 false-positive detected event, 0 missed injected events**; event precision **0.80**, event recall **1.00**. Matching is greedy with a one-to-one event-interval overlap criterion. This is a tiny constructed benchmark, not real-world accuracy. No generalization claim or production readiness claim is supported.

## Research basis
- UCSD DERConnect official capabilities: https://derconnect.ucsd.edu/testers/capabilities . Publicly listed *24 building meters*, *236 submeters*, and 2-second collection for building meters. Sampling cadence for *all* submeters is not established.
- SarSim0 (Oreshkin et al., arXiv 2601.00970) supports combining stochastic signals, seasonality and noise for synthetic time series. This prototype borrows only high-level **compositional design**; it does not implement SarSim0/SARIMA.
- Burgos et al. (Applied Sciences 14, 1194, 2024): consumption anomalies and cybersecurity abnormalities are distinct; an unusual demand value is not proof of attack.
- Lai et al. (NeurIPS 2021): distinguish point and pattern behavior; the classifications in this prototype are descriptive, not causes.
- Si et al. (TimeSeriesBench, arXiv 2402.10802): event-level metrics and false-alert burden are relevant for operator workflows.

## What changes with real DERConnect data
1. Authorize meter readings and publish only approved anonymized/aggregated materials; no public operational data or credentials.
2. Confirm meter identifiers, hierarchy, timestamp timezone, interval semantics (instantaneous vs average), per-device rates, units (kW vs kWh), direction of flow and sign conventions.
3. Replace CSV loading through a typed loader/adapter; normalize to observation schema; validate timestamp, quality and completeness.
4. Obtain actual weather and occupancy/schedule data; drop simulated schedule/oracle knowledge, learn baseline coefficients and uncertainty from historical training only.
5. Validate meter hierarchy (avoid adding parent and child), handle irregular intervals and meter outages properly, and preserve DST-aware timestamps.
6. Validate anomaly labels with operators; evaluate against real competing non-fault events, different seasons and held-out meters.
7. Make role-based secure access and deployment review a precondition for any campus-connected application.

## Manual user-test script
Ask a reviewer (without consulting `ground_truth.json`) to: select the highest-priority meter; explain in one sentence why it was flagged; state estimated flagged kWh and its conditional nature; identify two alternative explanations; and decide whether to open an investigation. Check correctness against the rendered data and capture uncertain decisions as feedback.

## Known limitations
Frontend build and visual QA not completed due to npm dependency installation timeout. Dates are in UTC; real operator displays should support a campus-local timezone. Ground-truth labels and synthetic meter properties are intentionally fictional. The fixed weather adjustment can be misspecified; no uncertainty interval is calibrated. This prototype does not test FDI attacks, grid physics, equipment degradation, or generalization to UCSD meters.
