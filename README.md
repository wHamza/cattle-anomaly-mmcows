# Cattle Behaviour Anomaly Detection (MMCOWS)

Detecting lethargy-like behaviour changes in dairy cows from **hourly** wearable-sensor data, **without labels**, and measuring honestly how well it works.

The hourly granularity is deliberate: it mirrors a LoRaWAN neck collar that reports once per hour, so the pipeline works under the same constraint a real farm deployment would have.

## Problem

A farmer wants a short daily list of cows to check. Too many false alarms and the system gets ignored; missed problems cost milk and animal welfare. The dataset has **no illness or heat labels** inside the recording window, so the system must learn what "normal" looks like for each cow and flag sustained deviations.

## Data

[MMCOWS](https://github.com/neis-lab/mmcows) (NeurIPS 2024 Datasets and Benchmarks): 10 instrumented Holstein cows, 14 days, one free-stall pen in Wisconsin.

| Sensor | Rate | Used as |
|---|---|---|
| Ankle sensor: lying / standing | 1 min | model input |
| Neck UWB tag: 3D position | 15 s | model input (movement, neck height) |
| Core body temperature | 1 min | **validation only** |
| Barn temperature-humidity index (THI) | 1 min | context |

The data is **not included** in this repo (large, and distributed under a non-commercial licence). The notebook downloads only the ~60 MB of sensor files it needs from [Hugging Face](https://huggingface.co/datasets/neis-lab/mmcows) without pulling the 18 GB archive.

## Approach

1. **Hourly features per cow:** lying fraction, distance moved, mean neck height and its variability.
2. **Quality filter:** drop hours with incomplete sensor coverage.
3. **Herd-relative deviation:** subtract the herd median for each hour. This removes everything that affects all cows at once: heat, milking, feeding, day/night rhythm.
4. **Per-cow robust z-score:** scale each cow's deviation by her own median and MAD, so active and calm cows become comparable.
5. **Directional lethargy score:** more lying + less movement + lower head. A first version using absolute z-scores had **no** relationship with the validation signal, because it treated "lying much more" and "lying much less" as the same thing.
6. **Smoothing and alarm rule:** rolling mean over the last 6 hours; alarm when it stays above the threshold for at least 3 consecutive hours.
7. **Threshold from an alarm budget:** about **one event per cow per week**, instead of an arbitrary cut-off.

Cow ↔ neck-tag IDs were not documented, so the mapping was **verified from the data**: for the right pair, the neck drops sharply whenever the ankle sensor reports lying.

## Evaluating without labels

- **Proxy signal:** herd-relative core body temperature. The lethargy score rises with it across deciles (−0.39 → +0.45 z) and the relationship is positive in **10 of 10 cows** (Spearman 0.14–0.48).
- **Synthetic drills:** a cow is made artificially lethargic for 24 h (lying +0.3, movement ×0.6) and the unchanged pipeline, with a fixed threshold, has to catch it. 80 drills per setting (10 cows × 8 start times). Drills that overlap a natural alarm are reported separately so they do not inflate the detection rate.

## Results

| Smoothing window | Alarm events / cow / week | Detection (clean drills) | Median delay | Contaminated drills |
|---|---|---|---|---|
| 3 h | 0.96 | 70% | 8 h | 16% |
| **6 h** | **1.02** | **84%** | **9 h** | 16% |
| 12 h | 1.02 | 84% | 10 h | 24% |

Each window uses its own threshold chosen with the same alarm budget. The 3-hour window is noisier and catches less at the same budget; the 12-hour window catches the same share but later, and its longer alarms overlap more with natural ones. **6 hours is the best trade-off.**

Also found: every cow loses sensor coverage at **04:00 and 15:00**, the milking times, when cows leave the pen. A real deployment needs coverage in the milking parlour or must treat those hours as expected gaps.

## Limitations

- 10 cows over 14 days, with no real illness or heat events to validate against.
- Core temperature is a proxy, and it informed the direction of the score, so it is not a fully independent test. Lying down can itself raise body temperature.
- The synthetic drill changes lying and movement only; real lethargy would also lower neck height. Adding lying time to a cow that is already lying at night has no effect (capped at 1).
- UWB position jitter inflates movement (~15 m/h even for a still cow).

## Run it

Open `cattle_anomaly.ipynb` in Google Colab and run all cells. The notebook is in Turkish, the code in English. Sections 4–8 only need the generated `hourly.parquet`.

```bash
pip install -r requirements.txt
```

## Next steps

- Compare with Isolation Forest and an autoencoder at the same alarm budget.
- Vary drill severity to draw a detection-vs-severity curve.
- Smooth UWB positions before computing movement.
- Move to data from the LoRaWAN collar system I am building, where heat events can be labelled.

## Citation

Data: Vu et al., *MMCOWS: A Multimodal Dataset for Dairy Cattle Monitoring*, NeurIPS 2024 Datasets and Benchmarks Track.
