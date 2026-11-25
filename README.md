# Drug Dosage Model

Computational Science final project by Balagao, Insigne, and Osorio

This repository contains the final Python program and selected outputs from the verified run. Use `reference_results/` to write the Discussion of Results and the Validation, Analysis and Reporting sections. The schedule experiments use the hypothetical handout drug. The Theoph comparison evaluates single-dose predictions against measured theophylline concentrations.

## Start with these results

| Manuscript topic | Data and figures |
| --- | --- |
| Baseline over 48 hours | [Summary](reference_results/baseline_summary.csv) · [Concentration plot](reference_results/fig2_baseline_48h.png) |
| Effects of daily dose and frequency | [Frequency comparison](reference_results/rq2_frequency_6000mg.csv) · [Dose comparison](reference_results/rq2_dosage_4_per_day.csv) · [Plots](reference_results/fig3_rq2_frequency_and_dosage.png) |
| Schedules that meet the target | [Candidates at 4.60 L](reference_results/rq3_candidates_4.60L.csv) · [Selected schedules at all three volumes](reference_results/rq4_selected_schedules.csv) |
| Blood-volume and body-weight effects | [Same schedule at three volumes](reference_results/rq4_same_schedule_three_volumes.csv) · [Weight examples](reference_results/option3_dosing_chart_by_weight.csv) · [Volume plots](reference_results/fig5_rq4_blood_volume.png) |
| Numerical verification | [All 12 checks](reference_results/verification_checks.csv) · [Hand calculation](reference_results/verification_manual.csv) · [Time-step comparison](reference_results/verification_time_step.csv) |
| Sensitivity to the numerical step | [Changed classifications](reference_results/refinement_changes.csv) · [Estimated dose windows](reference_results/supplementary_estimated_windows.csv) |
| External comparison with Theoph | [Inputs and assumptions](reference_results/theoph_parameters.csv) · [Overall metrics](reference_results/theoph_overall_metrics.csv) · [Errors by subject](reference_results/theoph_subject_metrics.csv) · [Measured and predicted values](reference_results/theoph_observed_vs_predicted.csv) · [Concentration curves](reference_results/fig6_theoph_observed_vs_model.png) · [Prediction errors](reference_results/fig8_theoph_prediction_errors.png) |

For schedule comparisons, report both the criterion for the final repeating cycle and the stricter criterion for staying in range from first entry. Keep the program’s numerical checks separate from the comparison with real observations. The primary Theoph metrics use 120 post-dose observations and give an MAE of 1.3662 mg/L and an RMSE of 1.8607 mg/L. The 12 time-zero observations supply initial conditions.

## Reproduce the run

Use Python 3.14. The reference outputs came from Python 3.14.7 with the dependency versions in `requirements.txt`. Run the commands from the repository root.

**macOS or Linux**

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python drug_dosage_model.py --no-ui --out results
```

**Windows PowerShell**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe drug_dosage_model.py --no-ui --out results
```

The run writes fresh tables and plots to `results/`. To open the local calculator afterward, run the same command without `--no-ui`.

The program reads both `data/Theoph.csv` and `data/Theoph_provenance.json`. The [official R dataset documentation](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/Theoph.html) describes the published data and the original study.
