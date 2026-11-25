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

The run writes fresh tables and plots to `results/`. Run the same command without `--no-ui` to open the local calculator. If it does not open automatically, use the address printed in the terminal. Select **Stop calculator** or press Ctrl+C to end the session.

Keep `web/` next to `drug_dosage_model.py`. It contains `index.html`, `calculator.css`, and `calculator.js`. Start the browser through Python so the page can call the simulation and load the current run’s results.

In VS Code, select the virtual environment as the Python interpreter and use **Run Python File**. The optional Interactive Window requires `ipykernel` in the same environment.

The program reads both `data/Theoph.csv` and `data/Theoph_provenance.json`. The [official R dataset documentation](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/Theoph.html) describes the published data and the original study.

## Model and assumptions

Chapter 14 supplies the one-compartment model with oral absorption [1]. The
48-hour baseline uses 6,000 mg once per day, a 4.60 L volume, an absorption
rate of 0.25 per hour, and a 6-hour half-life. The modeled target is at least
800 mg/L and below 1,000 mg/L. The extended experiments run for 168 hours at
blood volumes of 3.68, 4.60, and 5.52 L.

The program uses forward Euler [3]. Both transfers are calculated from the
amounts at the beginning of the time step.

```text
ke = ln(2) / half_life
absorbed = dt * ka * gut
eliminated = dt * ke * plasma
next_gut = gut - absorbed + intake
next_plasma = plasma + absorbed - eliminated
concentration = next_plasma / volume
```

Doses are equally spaced at intervals of `24 / doses_per_day` hours, starting
at time zero. The modeled intake is the dose multiplied by bioavailability [2].
It enters the gut at the end of the step and becomes available for absorption
at the next step. A dose at the final time boundary is excluded. The program
tracks cumulative elimination to check the amount balance. For example, at
0.02 hours the baseline hand calculation gives 5,985 mg in the gut, 15 mg in
plasma, and 3.26087 mg/L.

The weight-and-age calculation multiplies body weight by an estimated blood
volume per kg. It uses 0.080 L/kg below age 1, 0.075 L/kg from age 1 to below
12, and 0.070 L/kg from age 12. The coefficients are within published
reference ranges [4]. The age cutoffs are assumptions of this simulation.

## Published-data comparison

Part 9b uses the R Theoph data and published theophylline parameters [5, 6].
It uses each subject's recorded dose and weight and compares the predictions
at the observed times. Here the volume is the label's apparent volume of
distribution. The program uses recorded weight as a proxy for ideal body
weight because the dataset lacks the latter. It estimates the absorption rate
from the assumed time of peak using the first-order one-dose model [2].

Part 9 contains a separate optional regimen comparison. `REAL_DRUG` is set
to `None` in the distributed program. The generated
`option4_values_to_look_up.csv` lists the configuration fields and their
units. It accepts a time of peak or an absorption rate, plus optional
age-group parameter sets and measured concentrations. Part 9b runs
independently of this configuration.

## Sources

1. Gordon, S. I., and Guilfoos, B. (2017). *Introduction to Modeling and Simulation with MATLAB and Python*. Chapman and Hall/CRC. [Chapter 14 companion material](https://www.intromodeling.com/introduction-to-modeling-and-simulation-with-matlab-and-python/chapter-14/).
2. Dhillon, S., and Gill, K. (2006). Basic pharmacokinetics. In *Clinical Pharmacokinetics* (pp. 1–44). Pharmaceutical Press.
3. Brorson, S. [Forward Euler method](https://math.libretexts.org/Bookshelves/Differential_Equations/Numerically_Solving_Ordinary_Differential_Equations_%28Brorson%29/02%3A_Forward_Euler_method). Mathematics LibreTexts.
4. University of Iowa Carver College of Medicine. [Circulating blood volume table](https://iowaprotocols.medicine.uiowa.edu/protocols/maximum-allowable-blood-loss) in *Maximum Allowable Blood Loss*.
5. Chartwell RX, LLC. (2026). [Theophylline Oral Solution, USP](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=1b685384-5d96-46fa-b93b-e96538a2cfe7). DailyMed prescribing information, Clinical Pharmacology.
6. R Core Team. [Theoph data and original study attribution](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/Theoph.html). R datasets package.
