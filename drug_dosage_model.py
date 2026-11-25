"""Drug Dosage Model (Balagao, Insigne, Osorio).

HOW TO RUN IN VS CODE: install Python 3 and the Python and Jupyter extensions,
then type this in the VS Code terminal (on Windows, "py" instead of "python3"):
    python3 -m pip install pandas matplotlib ipykernel
Press "Run Python File" (under a minute): the tables print in the terminal,
and every table (.csv) and graph (.png) is saved in a results folder for the
report (its full path is printed at the end). Right-click in this file > "Run
Current File in Interactive Window" also works: tables and graphs then appear
like a notebook. The whole file also runs as one cell of a Jupyter notebook.

At the end (PART 11) the calculator web application opens in the browser at
http://127.0.0.1:8000. Its tabs run the study's calculations: Patient dosage
(RQ5), Try a schedule (RQ1), Compare schedules (RQ2, RQ4), Find schedules (RQ3,
RQ4), Measured data (RQ6) and Study results. Press "Stop calculator" on the
page (or Ctrl+C in the terminal) to end the program. No extra installs:
Python's own web server runs the page on this computer only. WEB_APP = False
types the age and weight under the output instead (Enter = 30 years, 70 kg).

A REAL DATASET: PART 9b compares the model with all 132 measured theophylline
concentrations in data/Theoph.csv, converted from R's official Theoph dataset.
It uses published label parameters without fitting them to these observations.
Keep Theoph.csv and Theoph_provenance.json together in the data folder.
Run without waiting for the calculator:
    python3 drug_dosage_model.py --no-ui --out results_check

A CUSTOM REAL DRUG REGIMEN (handout option 4): fill in REAL_DRUG at the start of PART 9 with
values from the drug's prescribing information and run again. Until then,
PART 9 only lists the values to look up.

Sources
[1] Drug Dosage Model Project handout, Chapter 14 companion site of Gordon S. I.,
    Guilfoos B. 2017. Introduction to Modeling and Simulation with MATLAB and
    Python. CRC Press.
    http://www.intromodeling.com/wp-content/uploads/2021/11/DRUG-DOSAGE-MODEL-PROJECT.pdf
[2] Dhillon S, Gill K. 2006. Basic pharmacokinetics. In Clinical
    Pharmacokinetics, Pharmaceutical Press, pp. 1-44, especially pp. 5, 9-14.
[3] Brorson S. Forward Euler method. Numerically Solving Ordinary Differential
    Equations, Mathematics LibreTexts.
[4] University of Iowa Carver College of Medicine. Maximum Allowable Blood
    Loss, Circulating Blood Volume table. The model uses 70, 75 and 80 mL/kg
    within the adult, child and infant reference ranges. The age cutoffs at
    1 and 12 years are simulation choices.
    https://iowaprotocols.medicine.uiowa.edu/protocols/maximum-allowable-blood-loss
[5] Chartwell RX, LLC. Theophylline Oral Solution, USP. DailyMed prescribing
    information, Clinical Pharmacology, revised February 2026 (PART 9b).
    https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=1b685384-5d96-46fa-b93b-e96538a2cfe7

Handout equations [1]; i = time step number, dt = time step (hours):
    absorption(i) = dt * absorption_rate * medicine_in_intestines(i)
    excretion(i)  = dt * excretion_rate * plasma_level(i)
    medicine_in_intestines(i+1) = medicine_in_intestines(i) - absorption(i) + intake(i)
    plasma_level(i+1) = plasma_level(i) - excretion(i) + absorption(i)
    plasma_concentration(i+1) = plasma_level(i+1) / blood_volume

Adjustments needed before the handout formula can run:
  * excretion_rate is not given. First-order elimination with a 6 h half-life
    gives excretion_rate = ln(2) / half_life = 0.1155 per hour [2].
  * intake(i) is not defined. It is one dose (daily total / doses per day)
    every 24 / doses_per_day hours starting at t = 0, and 0 at other steps.
  * eliminated(i+1) = eliminated(i) + excretion(i) is added only to check the
    drug balance: total swallowed = intestines + plasma + eliminated.
  * Typing errors fixed: the stray "(" in the excretion line, and
    plasma_level(I + 1) -> plasma_level(i + 1).
  * bioavailability (fraction of a dose that reaches the blood [2]) is added
    for the real drug: intake(i) = bioavailability x dose. The handout's
    equations absorb the whole dose, so it is 1 there and every handout
    result is unchanged.

The handout's optional additions [1] in this program:
  * Option 2 (PART 6): blood volumes 20% smaller and 20% larger.
  * Option 3 (PARTS 8 and 11): blood volume = 0.07 L/kg x body weight [4]
    (more per kg for children); the program searches the dosing schedules for
    a patient's age and weight, on a calculator web page.
  * Option 4 (PART 9): a real drug's published values replace the handout's;
    its volume is the volume of distribution from the label.
"""

# %% PART 0. Setup and parameters
import argparse
import base64
import html
import hashlib
import io
import json
import math
import os
import sys
import threading
import webbrowser
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib

if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')    # script run: no graph windows; the graphs are shown on the web page (PART 11)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

LOW, HIGH = 800.0, 1000.0              # medicinal and toxic levels, mg/L [1]
FREQUENCIES = (1, 2, 3, 4, 6, 8, 12, 24)
TOTAL_STEP = 500                       # mg/day between tested daily totals
DAILY_TOTALS = tuple(range(6000, 18001, TOTAL_STEP))
VOLUMES = (3.68, 4.60, 5.52)           # 20% smaller, baseline, 20% larger [1]
BLOOD_L_PER_KG = 0.07                  # adult reference coefficient [4]
# Exclusive age cutoffs (years) chosen for this simulation and coefficients (L/kg) from [4].
BLOOD_L_PER_KG_BY_AGE = ((1, 0.080), (12, 0.075), (math.inf, BLOOD_L_PER_KG))
WEIGHTS = (50, 60, 70, 80, 90, 100)    # body weights in the option 3 dosing chart, kg
EXTENDED_H = 168.0                     # one week: long enough to repeat
REFINED_DT = 0.005                     # finer step used to re-check candidates
OUT = 'results'                        # folder for the .csv tables and .png graphs
WEB_APP = True                         # PART 11 calculator web page; False: type age and weight under the output
WEB_PORT = 8000                        # the page is http://127.0.0.1:8000 (another port if 8000 is busy)
NO_UI = False                         # --no-ui runs the complete study without waiting for a patient
if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    parser = argparse.ArgumentParser(description='Run the drug dosage study and published-data comparison.')
    parser.add_argument('--no-ui', action='store_true', help='Save the study without opening the calculator.')
    parser.add_argument('--out', default=OUT, help='Folder for tables, graphs and the validation report.')
    command_line = parser.parse_args()
    NO_UI, OUT = command_line.no_ui, command_line.out


def load_theoph_dataset():
    """Read the bundled data and verify its documented source conversion."""
    if '__file__' in globals():
        data_dir = Path(__file__).resolve().parent / 'data'
    else:
        data_dir = Path.cwd() / 'data'
        if not data_dir.exists():
            data_dir = Path.cwd() / 'FinalProject' / 'data'
    data_path = data_dir / 'Theoph.csv'
    provenance_path = data_dir / 'Theoph_provenance.json'
    try:
        content = data_path.read_bytes()
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as problem:
        raise SystemExit(f'Theoph comparison cannot start. Keep Theoph.csv and '
                         f'Theoph_provenance.json from the submission together in {data_dir}.\n{problem}') from None
    if not isinstance(provenance, dict) or hashlib.sha256(content).hexdigest() != provenance.get('csv_sha256'):
        raise SystemExit('Theoph.csv does not match its documented checksum. Restore both files from the submission.')
    measured = pd.read_csv(io.BytesIO(content))
    if list(measured.columns) != ['Subject', 'Wt', 'Dose', 'Time', 'conc'] or len(measured) != 132:
        raise ValueError('Unexpected Theoph columns or number of measurements.')
    if measured.isna().any().any() or measured.groupby('Subject').size().to_dict() != dict.fromkeys(range(1, 13), 11):
        raise ValueError('Expected 12 subjects with 11 complete measurements each.')
    return measured, provenance


theoph_data, theoph_provenance = load_theoph_dataset()
os.makedirs(OUT, exist_ok=True)

# Colorblind-safe series colors; green and red mark the two concentration limits.
COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')
GREEN, RED, INK = '#0ca30c', '#d03b3b', '#52514e'
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200, 'axes.grid': True,
                     'grid.color': '#e1e0d9', 'grid.linewidth': 0.8,
                     'axes.edgecolor': '#c3c2b7', 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 2,
                     'legend.frameon': False, 'legend.fontsize': 8})
pd.set_option('display.width', 250)


@dataclass(frozen=True)
class Parameters:
    daily_mg: float = 6000.0
    doses_per_day: int = 1
    volume_l: float = 4.60
    ka_per_h: float = 0.25
    half_life_h: float = 6.0
    dt_h: float = 0.01
    duration_h: float = 48.0
    bioavailability: float = 1.0       # fraction of each dose that reaches the blood [2]
    medicinal_mg_l: float = LOW
    toxic_mg_l: float = HIGH
    single_dose: bool = False         # published single-dose studies must not receive another dose at 24 h

    @property
    def ke_per_h(self):
        # First-order elimination relationship from [2], p. 11.
        return math.log(2.0) / self.half_life_h


# %% PART 0b. The model (handout equations) and the analysis functions
def integer_steps(hours, dt):
    steps = round(hours / dt)
    if steps < 1 or not math.isclose(steps * dt, hours, abs_tol=1e-9):
        raise ValueError('Duration and dose interval must be multiples of dt.')
    return steps


def simulate(p):
    """Literal handout update order: old-state fluxes, then intake [1].

    intake[i] is scheduled at i*dt and enters gut[i+1]. Thus a dose is
    available to absorption at the next boundary, a one-step lag. Doses
    scheduled exactly at duration_h are excluded. intake is the part of a
    dose that is absorbed, bioavailability x dose (all of it in the handout).
    """
    values = (p.daily_mg, p.volume_l, p.ka_per_h, p.half_life_h, p.dt_h,
              p.duration_h, p.bioavailability, p.medicinal_mg_l, p.toxic_mg_l)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('All numerical parameters must be finite.')
    if p.daily_mg < 0 or p.volume_l <= 0 or p.ka_per_h < 0:
        raise ValueError('Require dosage >= 0, volume > 0, ka >= 0.')
    if min(p.half_life_h, p.dt_h, p.duration_h) <= 0:
        raise ValueError('Half-life, dt, and duration must be positive.')
    if not 0 < p.bioavailability <= 1:
        raise ValueError('Bioavailability must be above 0 and at most 1.')
    if not 0 < p.medicinal_mg_l < p.toxic_mg_l:
        raise ValueError('Require 0 < medicinal level < toxic level.')
    if not isinstance(p.doses_per_day, int) or p.doses_per_day < 1:
        raise ValueError('doses_per_day must be a positive integer.')
    if p.dt_h * max(p.ka_per_h, p.ke_per_h) > 1:
        raise ValueError('Time step permits negative drug amounts; reduce dt.')
    n = integer_steps(p.duration_h, p.dt_h)
    spacing = integer_steps(24.0 / p.doses_per_day, p.dt_h)
    t = [i * p.dt_h for i in range(n + 1)]
    gut, plasma, eliminated, total_in = ([0.0] * (n + 1) for _ in range(4))
    intake = [0.0] * n
    dose = p.bioavailability * p.daily_mg / p.doses_per_day
    for i in range(n):
        scheduled = i == 0 if p.single_dose else i % spacing == 0
        intake[i] = dose if scheduled else 0.0
        absorbed = p.dt_h * p.ka_per_h * gut[i]           # absorption(i)
        excreted = p.dt_h * p.ke_per_h * plasma[i]        # excretion(i)
        gut[i + 1] = gut[i] - absorbed + intake[i]
        plasma[i + 1] = plasma[i] - excreted + absorbed
        eliminated[i + 1] = eliminated[i] + excreted
        total_in[i + 1] = total_in[i] + intake[i]
    concentration = [amount / p.volume_l for amount in plasma]
    return dict(p=p, time=t, gut=gut, plasma=plasma, eliminated=eliminated,
                total_in=total_in, intake=intake, concentration=concentration)


def first_entry(t, c, low, high):
    """First entry into [low, high), interpolated along numerical segments."""
    if low <= c[0] < high:
        return t[0]
    for i in range(len(c) - 1):
        if c[i] < low <= c[i + 1]:
            return t[i] + (t[i + 1] - t[i]) * (low - c[i]) / (c[i + 1] - c[i])
        if c[i] >= high > c[i + 1]:
            # Infimum of entry times on a decreasing upper-threshold crossing.
            return t[i] + (t[i + 1] - t[i]) * (high - c[i]) / (c[i + 1] - c[i])
    return None


def exposure_hours(t, c, low, high):
    """Hours below, within, and at/above limits; piecewise-linear estimate.

    Split every time segment at threshold crossings. No point count is
    mistaken for elapsed time. Isolated threshold contacts have zero duration.
    """
    totals = [0.0, 0.0, 0.0]
    for i in range(len(t) - 1):
        a, b = c[i], c[i + 1]
        cuts = [0.0, 1.0]
        if b != a:
            cuts.extend(f for level in (low, high)
                        if 0 < (f := (level - a) / (b - a)) < 1)
        cuts.sort()
        for left, right in zip(cuts, cuts[1:]):
            midpoint = a + (b - a) * (left + right) / 2
            group = 0 if midpoint < low else (1 if midpoint < high else 2)
            totals[group] += (right - left) * (t[i + 1] - t[i])
    return totals


def summarize(run):
    p, t, c = run['p'], run['time'], run['concentration']
    low, high = p.medicinal_mg_l, p.toxic_mg_l
    entry = first_entry(t, c, low, high)
    below, within, above = exposure_hours(t, c, low, high)
    after_ok = False
    enough_followup = False
    if entry is not None:
        # Entry occurs at the medicinal level for the default zero-drug initial condition.
        remaining = [low] + [v for ti, v in zip(t, c) if ti >= entry]
        enough_followup = p.duration_h - entry >= 24.0 / p.doses_per_day
        after_ok = enough_followup and min(remaining) >= low and max(remaining) < high
    m = integer_steps(24.0 / p.doses_per_day, p.dt_h)
    cycle_change, cycle_min, cycle_max, cycle_mean, cycle_ok = None, None, None, None, False
    if len(c) >= 2 * m + 1:
        current, previous = c[-m - 1:], c[-2 * m - 1:-m]
        scale = max(current) if max(current) > 0 else 1.0    # 0/0 for a zero dose
        cycle_change = max(abs(a - b) for a, b in zip(current, previous)) / scale
        cycle_min, cycle_max = min(current), max(current)
        cycle_mean = sum(current[1:]) / m
        cycle_ok = cycle_change < 0.001 and cycle_min >= low and cycle_max < high
    balance = max(abs(g + q + e - d) for g, q, e, d in
                  zip(run['gut'], run['plasma'], run['eliminated'], run['total_in']))
    return dict(daily_mg=p.daily_mg, doses_per_day=p.doses_per_day,
                mg_per_dose=p.daily_mg / p.doses_per_day,
                interval_h=24.0 / p.doses_per_day, volume_l=p.volume_l,
                duration_h=p.duration_h, dt_h=p.dt_h,
                peak_full_mg_l=max(c), trough_full_mg_l=min(c),
                first_entry_h=entry, below_h=below, within_h=within,
                at_or_above_toxic_h=above,
                full_run_ever_reaches_toxic=max(c) >= high,
                at_least_one_interval_after_entry=enough_followup,
                stays_in_range_after_first_entry=after_ok,
                last_cycle_min_mg_l=cycle_min, last_cycle_max_mg_l=cycle_max,
                last_cycle_mean_mg_l=cycle_mean,
                last_cycle_relative_change=cycle_change,
                last_cycle_converged=cycle_change is not None and cycle_change < 0.001,
                last_cycle_in_range_and_converged=cycle_ok,
                max_mass_balance_error_mg=balance,
                medicinal_mg_l=low, toxic_mg_l=high)


def exact_concentration(p, time_points):
    """Independent continuous solution, derived from this linear model.

    The absorbed part of a dose (bioavailability x dose) enters the gut at
    scheduled time + dt, matching the handout's right-boundary intake
    convention. Summation uses linearity. The result verifies numerical
    implementation, not clinical validity.
    """
    ka, ke = p.ka_per_h, p.ke_per_h
    interval, dose = 24.0 / p.doses_per_day, p.bioavailability * p.daily_mg / p.doses_per_day
    count = 1 if p.single_dose else math.ceil(p.duration_h / interval)
    result = []
    for t in time_points:
        amount = 0.0
        for j in range(count):
            elapsed = t - (j * interval + p.dt_h)
            if elapsed > 0:
                if math.isclose(ka, ke, rel_tol=1e-12, abs_tol=1e-12):
                    amount += dose * ka * elapsed * math.exp(-ke * elapsed)
                else:
                    amount += dose * ka / (ka - ke) * (math.exp(-ke * elapsed) - math.exp(-ka * elapsed))
        result.append(amount / p.volume_l)
    return result


def ka_from_peak_time(tmax_h, half_life_h):
    """Absorption rate whose single-dose peak comes at tmax_h (labels give Tmax, not ka).

    Setting dC/dt = 0 in the exact single-dose solution above gives
    tmax = ln(ka / ke) / (ka - ke). For ka > ke this falls from 1/ke towards 0,
    so halving the search interval (bisection) finds the one ka > ke with that tmax.
    """
    ke = math.log(2.0) / half_life_h
    if not 0 < tmax_h < 1 / ke:
        raise ValueError(f'Time of peak must be between 0 and {1 / ke:.2f} h for this '
                         'half-life; otherwise enter ka_per_h instead.')

    def peak_time(ka):
        return math.log(ka / ke) / (ka - ke)

    low, high = ke, 2 * ke
    while peak_time(high) > tmax_h:
        low, high = high, 2 * high
    for _ in range(100):
        middle = (low + high) / 2
        if peak_time(middle) > tmax_h:
            low = middle
        else:
            high = middle
    return (low + high) / 2


# %% PART 0c. Helpers for tables and graphs
def heading(text):
    print('\n' + '=' * 80 + f'\n{text}\n' + '=' * 80)


def show(title, rows, csv_name=None):
    """Print a list of dicts as a table and save it in OUT as CSV."""
    table = pd.DataFrame(rows)
    if csv_name:
        table.to_csv(os.path.join(OUT, csv_name), index=False)
    print(f'\n{title}\n{table.to_string(index=False)}')
    return table


def yes_no(flag):
    return 'Yes' if flag else 'No'


def hours(value):
    return 'never' if value is None else round(value, 2)


def zone(c, low=LOW, high=HIGH):
    return f'below {low:g}' if c < low else ('in range' if c < high else 'toxic')


def sig(value, digits=4):
    """Round to significant digits, for concentrations of any size."""
    return float(f'{value:.{digits}g}')


def schedule_text(r):
    """'11,500 mg/day as 4 doses/day' for a summary row, or 'no tested schedule'."""
    if r is None:
        return 'no tested schedule'
    n = r['doses_per_day']
    return f"{r['daily_mg']:,g} mg/day as {n} dose{'s' if n > 1 else ''}/day"


def status(row):
    """Short label for the final repeating cycle of a run (see Table 9 legend)."""
    if not row['last_cycle_converged']:
        return 'n/c'
    if row['last_cycle_in_range_and_converged']:
        return 'OK+' if row['stays_in_range_after_first_entry'] else 'OK'
    if row['last_cycle_max_mg_l'] < row['medicinal_mg_l']:
        return 'low'
    if row['last_cycle_min_mg_l'] >= row['medicinal_mg_l']:
        return 'high'
    return 'dip' if row['last_cycle_max_mg_l'] < row['toxic_mg_l'] else 'swing'


def lookup(rows, daily, frequency, volume):
    return next(r for r in rows if r['daily_mg'] == daily and r['doses_per_day'] == frequency
                and math.isclose(r['volume_l'], volume))


def plot_concentration(ax, series, title):
    """Concentration curves with the target range (medicinal to toxic level) shaded."""
    p = series[0][1]['p']
    low, high = p.medicinal_mg_l, p.toxic_mg_l
    end = max(run['time'][-1] for _, run in series)
    top = max(max(run['concentration']) for _, run in series)
    ax.axhspan(low, high, color=GREEN, alpha=0.08, lw=0)
    ax.axhline(low, color=GREEN, lw=1.2, ls='--')
    ax.axhline(high, color=RED, lw=1.2, ls='--')
    # Limit labels sit just outside the right edge so they never cover a curve.
    ax.text(1.01, low, f'medicinal\n{low:g} mg/L', transform=ax.get_yaxis_transform(),
            va='center', color=INK, fontsize=8)
    ax.text(1.01, high, f'toxic\n{high:g} mg/L', transform=ax.get_yaxis_transform(),
            va='center', color=INK, fontsize=8)
    for color, (label, run) in zip(COLORS, series):
        ax.plot(run['time'], run['concentration'], color=color, label=label)
    ax.set(title=title, xlabel='Time (hours)', ylabel='Concentration (mg/L)',
           xlim=(0, end), ylim=(0, max(high * 1.12, top * 1.05)))
    if len(series) > 1:
        ax.legend(loc='best')


def plot_amounts(ax, run, title):
    """How much drug is in each part of the model: G, P and E."""
    for color, key, label in zip(COLORS, ('gut', 'plasma', 'eliminated'),
                                 ('Medicine in intestines G', 'Drug in plasma P', 'Eliminated so far E')):
        ax.plot(run['time'], run[key], color=color, label=label)
    ax.set(title=title, xlabel='Time (hours)', ylabel='Amount (mg)', xlim=(0, run['p'].duration_h), ylim=(0, None))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.legend(loc='upper left')


GRAPHS = []     # graph files saved by this run, also shown on the calculator web page (PART 11)


def save_figure(fig, name, layout_rect=None):
    fig.tight_layout(rect=layout_rect)
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    if name not in GRAPHS:
        GRAPHS.append(name)
    if 'ipykernel' in sys.modules:
        plt.show()      # notebook: the graph appears under the cell now
    else:
        plt.close(fig)  # script run: the graph is shown on the calculator web page (PART 11)


heading('PART 0. Model parameters and experiment design')
base = Parameters()
show('Table P1. Parameters (handout [1]; excretion rate derived from the half-life [2])', [
    {'Parameter': 'Doses per day (baseline)', 'Value': base.doses_per_day, 'Unit': 'doses/day', 'Source': 'handout'},
    {'Parameter': 'Dosage per day (baseline)', 'Value': base.daily_mg, 'Unit': 'mg/day', 'Source': 'handout'},
    {'Parameter': 'Absorption rate constant', 'Value': base.ka_per_h, 'Unit': 'per hour', 'Source': 'handout'},
    {'Parameter': 'Half-life of the drug', 'Value': base.half_life_h, 'Unit': 'hours', 'Source': 'handout'},
    {'Parameter': 'Excretion rate = ln(2) / half-life', 'Value': round(base.ke_per_h, 5), 'Unit': 'per hour', 'Source': 'derived [2]'},
    {'Parameter': 'Bioavailability (fraction of a dose absorbed)', 'Value': base.bioavailability, 'Unit': 'fraction', 'Source': 'handout (whole dose absorbed)'},
    {'Parameter': 'Blood volume (baseline)', 'Value': base.volume_l, 'Unit': 'L', 'Source': 'handout'},
    {'Parameter': 'Blood volume per kg of body weight (option 3)', 'Value': f'{BLOOD_L_PER_KG} (under 12 y: 0.075, under 1 y: 0.08)', 'Unit': 'L/kg',
     'Source': 'reference ranges [4]; age cutoffs are simulation choices'},
    {'Parameter': 'Medicinal level', 'Value': LOW, 'Unit': 'mg/L', 'Source': 'handout'},
    {'Parameter': 'Toxic level', 'Value': HIGH, 'Unit': 'mg/L', 'Source': 'handout'},
    {'Parameter': 'Time step (refinement step)', 'Value': f'{base.dt_h} ({REFINED_DT})', 'Unit': 'hours', 'Source': 'study choice'},
    {'Parameter': 'Run length: baseline / extended', 'Value': f'{base.duration_h:g} / {EXTENDED_H:g}', 'Unit': 'hours', 'Source': 'handout / study choice'},
], 'table_parameters.csv')
show('Table P2. Experiment design', [
    {'Experiment': 'RQ1 baseline', 'Daily total (mg/day)': '6000', 'Doses/day': '1',
     'Blood volume (L)': '4.60', 'Hours': '48 and 168', 'Time step (h)': '0.01'},
    {'Experiment': 'RQ2 frequency effect', 'Daily total (mg/day)': '6000', 'Doses/day': '1, 2, 3, 4, 6, 8, 12, 24',
     'Blood volume (L)': '4.60', 'Hours': '168', 'Time step (h)': '0.01'},
    {'Experiment': 'RQ2 dosage effect', 'Daily total (mg/day)': '6000-18000', 'Doses/day': '4',
     'Blood volume (L)': '4.60', 'Hours': '168', 'Time step (h)': '0.01'},
    {'Experiment': 'RQ3 schedule grid', 'Daily total (mg/day)': '6000-18000, step 500', 'Doses/day': '1, 2, 3, 4, 6, 8, 12, 24',
     'Blood volume (L)': '4.60', 'Hours': '168', 'Time step (h)': '0.01; candidates 0.005'},
    {'Experiment': 'RQ4 blood volume', 'Daily total (mg/day)': '6000-18000, step 500', 'Doses/day': '1, 2, 3, 4, 6, 8, 12, 24',
     'Blood volume (L)': '3.68, 4.60, 5.52', 'Hours': '168', 'Time step (h)': '0.01; candidates 0.005'},
    {'Experiment': 'Option 3 body weight', 'Daily total (mg/day)': 'average in range, step 500', 'Doses/day': '1, 2, 3, 4, 6, 8, 12, 24',
     'Blood volume (L)': '0.07 x weight, 50-100 kg', 'Hours': '168', 'Time step (h)': '0.01; chosen 0.005'},
    {'Experiment': 'Option 4 real drug', 'Daily total (mg/day)': 'label regimen', 'Doses/day': 'label regimen',
     'Blood volume (L)': 'label Vd x weight', 'Hours': '12 half-lives, at least 168', 'Time step (h)': '0.01'},
], 'table_experiment_design.csv')


# %% PART 1. Verification: does the code solve the handout equations correctly?
heading('PART 1. VERIFICATION - does the code solve the handout equations correctly?')


def verify():
    """Implementation checks: hand calculation, balance, scaling, exact solution."""
    p = Parameters()
    run = simulate(p)
    checks = []

    def check(name, passed, result):
        checks.append({'Check': name, 'Result': result, 'Status': 'PASS' if passed else 'FAIL'})

    # Hand calculation of the first steps (baseline, dt = 0.01 h, ke = 0.11552/h):
    #   i=0: absorption = 0.01*0.25*0 = 0            G1 = 0 - 0 + 6000 = 6000   P1 = 0
    #   i=1: absorption = 0.01*0.25*6000 = 15        G2 = 6000 - 15 = 5985
    #        excretion  = 0.01*0.11552*0 = 0         P2 = 0 - 0 + 15 = 15
    #   i=2: absorption = 0.01*0.25*5985 = 14.9625   G3 = 5985 - 14.9625 = 5970.0375
    #        excretion  = 0.01*0.11552*15 = 0.01733  P3 = 15 - 0.01733 + 14.9625 = 29.94517
    #   C = P / 4.6:  C2 = 3.26087,  C3 = 6.50982 mg/L
    hand = {1: (6000.0, 0.0, 0.0), 2: (5985.0, 15.0, 3.26087),
            3: (5970.0375, 29.94517, 6.50982)}
    manual, worst = [], 0.0
    for i, (g, q, c) in hand.items():
        program = (run['gut'][i], run['plasma'][i], run['concentration'][i])
        worst = max(worst, *(abs(a - b) for a, b in zip(program, (g, q, c))))
        manual.append({'Step i': i, 'Time (h)': round(run['time'][i], 2),
                       'G by hand (mg)': g, 'G program (mg)': round(program[0], 5),
                       'P by hand (mg)': q, 'P program (mg)': round(program[1], 5),
                       'C by hand (mg/L)': c, 'C program (mg/L)': round(program[2], 5)})
    check('Manual calculation, steps 1-3', worst < 1e-4, f'largest difference {worst:.1e}')

    balance = summarize(run)['max_mass_balance_error_mg']
    check('Drug balance: swallowed = G + P + E', balance < 1e-7, f'largest error {balance:.1e} mg')
    check('No negative drug amounts', min(run['gut']) >= 0 and min(run['plasma']) >= 0,
          'G >= 0 and P >= 0 at every step')
    check('Zero dose gives zero concentration',
          max(simulate(Parameters(daily_mg=0))['concentration']) == 0, 'max C = 0')
    check('No absorption (ka = 0) gives zero concentration',
          max(simulate(Parameters(ka_per_h=0))['concentration']) == 0, 'max C = 0')
    doubled = simulate(Parameters(daily_mg=12000))['concentration']
    smaller = simulate(Parameters(volume_l=3.68))['concentration']
    halved = simulate(Parameters(bioavailability=0.5))['concentration']
    scale_error = max(abs(b - 2 * a) for a, b in zip(run['concentration'], doubled))
    volume_error = max(abs(b - 1.25 * a) for a, b in zip(run['concentration'], smaller))
    absorbed_error = max(abs(b - 0.5 * a) for a, b in zip(run['concentration'], halved))
    check('Double the dose -> double the concentration', scale_error < 1e-8,
          f'largest difference {scale_error:.1e}')
    check('20% smaller volume -> 1.25 x concentration', volume_error < 1e-8,
          f'largest difference {volume_error:.1e}')
    check('Half the bioavailability -> half the concentration', absorbed_error < 1e-8,
          f'largest difference {absorbed_error:.1e}')

    convergence, errors, peaks = [], [], []
    for dt in (0.04, 0.02, 0.01, 0.005):
        q = Parameters(dt_h=dt)
        numerical = simulate(q)
        exact = exact_concentration(q, numerical['time'])
        errors.append(max(abs(a - b) for a, b in zip(numerical['concentration'], exact)))
        peaks.append(max(numerical['concentration']))
        convergence.append({'Time step (h)': dt, 'Max error vs exact (mg/L)': round(errors[-1], 4),
                            'Peak concentration (mg/L)': round(peaks[-1], 3)})
    check('Error shrinks as the time step shrinks', all(b < a for a, b in zip(errors, errors[1:])),
          ' -> '.join(f'{e:.3f}' for e in errors) + ' mg/L')
    change = abs(peaks[-1] - peaks[-2]) / peaks[-1]
    check('Peak changes < 0.1% from dt 0.01 to 0.005 h', change < 0.001, f'{change:.3%}')

    one = simulate(Parameters(duration_h=24.0))        # a single dose: none at 24 h
    top = max(range(len(one['time'])), key=one['concentration'].__getitem__)
    peak_h = one['time'][top] - p.dt_h                 # minus the one-step intake lag
    ka_back = ka_from_peak_time(peak_h, p.half_life_h)
    check('Time of peak -> absorption rate (used for a real drug)',
          abs(ka_back - p.ka_per_h) < 0.01 * p.ka_per_h,
          f'peak at {peak_h:.2f} h gives ka = {ka_back:.4f} per hour')
    single = simulate(Parameters(single_dose=True, duration_h=25.0))
    check('Single-dose study: no second dose at 24 h',
          sum(value > 0 for value in single['intake']) == 1
          and math.isclose(single['total_in'][-1], p.daily_mg),
          'one dose only, including the observations after 24 h')
    return checks, manual, convergence, errors


checks, manual, convergence, errors = verify()
show('Table 1. Manual calculation vs program (baseline, first time steps)', manual,
     'verification_manual.csv')
show('Table 2. Time-step refinement: forward Euler vs exact solution (baseline, 48 h)',
     convergence, 'verification_time_step.csv')
show('Table 3. Verification checks', checks, 'verification_checks.csv')
all_passed = all(row['Status'] == 'PASS' for row in checks)
print('\nALL VERIFICATION CHECKS PASSED' if all_passed else '\nSOME CHECKS FAILED - see Table 3')

run = simulate(base)
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
left.plot(run['time'], exact_concentration(base, run['time']), color=COLORS[1], lw=5,
          alpha=0.45, label='Exact solution')
left.plot(run['time'], run['concentration'], color=COLORS[0], label='Simulation, dt = 0.01 h')
left.set(title='Baseline: simulation vs exact solution', xlabel='Time (hours)',
         ylabel='Concentration (mg/L)', xlim=(0, 48), ylim=(0, None))
left.legend(loc='lower right')
steps = [row['Time step (h)'] for row in convergence]
right.loglog(steps, errors, 'o-', color=COLORS[0], markersize=8)
right.minorticks_off()
right.set_xticks(steps, [f'{dt:g}' for dt in steps])
right.set_yticks(errors, [f'{e:.2f}' for e in errors])
right.set(title='Halving the time step halves the error', xlabel='Time step (hours)',
          ylabel='Max error vs exact (mg/L)')
save_figure(fig, 'fig1_verification.png')


# %% PART 2. RQ1 - baseline: 6000 mg once a day, 4.60 L, 48 h
heading('PART 2. RQ1 - Baseline: 6000 mg once a day, blood volume 4.60 L, 48 hours')
baseline = simulate(base)
base48 = summarize(baseline)
base168 = summarize(simulate(Parameters(duration_h=EXTENDED_H)))
pd.DataFrame({'time_h': baseline['time'], 'intake_mg': baseline['intake'] + [0.0],
              'intestines_G_mg': baseline['gut'], 'plasma_P_mg': baseline['plasma'],
              'eliminated_E_mg': baseline['eliminated'],
              'concentration_C_mg_l': baseline['concentration']}).to_csv(
    os.path.join(OUT, 'baseline_48h_every_step.csv'), index=False)

rows = []
for i in range(0, len(baseline['time']), integer_steps(2.0, base.dt_h)):
    c = baseline['concentration'][i]
    rows.append({'Time (h)': round(baseline['time'][i], 2),
                 'Dose taken (mg)': baseline['intake'][i] if i < len(baseline['intake']) else 0.0,
                 'Intestines G (mg)': round(baseline['gut'][i], 1),
                 'Plasma P (mg)': round(baseline['plasma'][i], 1),
                 'Eliminated E (mg)': round(baseline['eliminated'][i], 1),
                 'Concentration C (mg/L)': round(c, 1), 'Zone': zone(c)})
show('Table 4. Baseline every 2 hours (a dose taken at step i reaches the intestines at step i+1)',
     rows, 'baseline_48h_every_2h.csv')

c = baseline['concentration']
peak_index = max(range(len(c)), key=c.__getitem__)
show('Table 5. Baseline summary', [
    {'Quantity': 'Peak concentration (mg/L)', 'Value': round(c[peak_index], 1)},
    {'Quantity': 'Time of peak (h)', 'Value': round(baseline['time'][peak_index], 2)},
    {'Quantity': 'Concentration just before the 2nd dose, 24 h (mg/L)', 'Value': round(c[integer_steps(24.0, base.dt_h)], 1)},
    {'Quantity': 'Concentration at 48 h (mg/L)', 'Value': round(c[-1], 1)},
    {'Quantity': 'Hours below 800 / in range / toxic (of 48)',
     'Value': f"{base48['below_h']:.1f} / {base48['within_h']:.1f} / {base48['at_or_above_toxic_h']:.1f}"},
    {'Quantity': 'First entry into 800-1000 mg/L', 'Value': hours(base48['first_entry_h'])},
    {'Quantity': 'Day 2 repeats day 1 within 0.1%?',
     'Value': f"{yes_no(base48['last_cycle_converged'])} (change {base48['last_cycle_relative_change']:.1%})"},
    {'Quantity': '168 h run: repeating cycle min - max (mg/L)',
     'Value': f"{base168['last_cycle_min_mg_l']:.1f} - {base168['last_cycle_max_mg_l']:.1f}"},
    {'Quantity': '168 h run: repeats within 0.1%? / hours in range',
     'Value': f"{yes_no(base168['last_cycle_converged'])} / {base168['within_h']:.1f}"},
], 'baseline_summary.csv')

fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7.5))
plot_concentration(top, [('6000 mg once a day', baseline)],
                   'Baseline plasma concentration: 6000 mg once a day, 4.60 L')
plot_amounts(bottom, baseline, 'Drug amounts in each part of the model')
save_figure(fig, 'fig2_baseline_48h.png')


# %% PART 3. Experiment grid (used by RQ2, RQ3 and RQ4)
heading('PART 3. Experiment grid: 25 daily totals x 8 frequencies x 3 volumes, 168 h each')
print('Running 600 schedules, about half a minute ...')
grid = []
for volume in VOLUMES:
    for frequency in FREQUENCIES:
        for daily in DAILY_TOTALS:
            grid.append(summarize(simulate(Parameters(daily_mg=daily, doses_per_day=frequency,
                                                      volume_l=volume, duration_h=EXTENDED_H))))
pd.DataFrame(grid).to_csv(os.path.join(OUT, 'experiment_grid_168h.csv'), index=False)

# Schedules that pass at dt = 0.01 h are re-run at dt = 0.005 h; the finer result is kept.
refined = []
for row in grid:
    if row['last_cycle_in_range_and_converged'] or row['stays_in_range_after_first_entry']:
        fine = summarize(simulate(Parameters(daily_mg=row['daily_mg'], doses_per_day=row['doses_per_day'],
                                             volume_l=row['volume_l'], duration_h=EXTENDED_H,
                                             dt_h=REFINED_DT)))
        fine['first_entry_h_at_dt_0.01'] = row['first_entry_h']
        fine['stays_in_range_at_dt_0.01'] = row['stays_in_range_after_first_entry']
        fine['cycle_in_range_at_dt_0.01'] = row['last_cycle_in_range_and_converged']
        fine['same_result_at_dt_0.01'] = (
            fine['cycle_in_range_at_dt_0.01'] == fine['last_cycle_in_range_and_converged']
            and fine['stays_in_range_at_dt_0.01'] == fine['stays_in_range_after_first_entry'])
        refined.append(fine)
pd.DataFrame(refined).to_csv(os.path.join(OUT, 'candidate_refinement_168h.csv'), index=False)
changed = [r for r in refined if not r['same_result_at_dt_0.01']]
print(f'{len(grid)} schedules tested; {len(refined)} candidates re-run at dt = {REFINED_DT} h; '
      f'{len(refined) - len(changed)} of {len(refined)} give the same result as dt = 0.01 h.')
show('Table 6. Candidates whose result changed with the finer time step (the dt = 0.005 h result is used)', [
    {'Volume (L)': r['volume_l'], 'Daily total (mg)': r['daily_mg'], 'Doses/day': r['doses_per_day'],
     'First entry, dt 0.01 (h)': hours(r['first_entry_h_at_dt_0.01']),
     'First entry, dt 0.005 (h)': hours(r['first_entry_h']),
     'Stays in range, dt 0.01': yes_no(r['stays_in_range_at_dt_0.01']),
     'Stays in range, dt 0.005': yes_no(r['stays_in_range_after_first_entry']),
     'Repeating cycle in range, dt 0.01 / 0.005': yes_no(r['cycle_in_range_at_dt_0.01']) + ' / '
                                                  + yes_no(r['last_cycle_in_range_and_converged'])}
    for r in changed] or [{'Result': 'none - every candidate gives the same result'}], 'refinement_changes.csv')
if changed:
    same_cycle = all(r['cycle_in_range_at_dt_0.01'] == r['last_cycle_in_range_and_converged'] for r in changed)
    print('Borderline cases: during build-up a peak or trough lands within a fraction of a mg/L of 800 mg/L,\n'
          'so a tiny numerical difference decides "stays in range after first entry".'
          + (' The repeating-cycle result did not change.' if same_cycle else ''))


# %% PART 4. RQ2 - effects of daily dosage and dosing frequency
heading('PART 4. RQ2 - Effects of daily dosage and dosing frequency (4.60 L, 168 h)')
rows = []
for frequency in FREQUENCIES:
    r = lookup(grid, 6000, frequency, 4.60)
    rows.append({'Doses/day': frequency, 'Every (h)': r['interval_h'], 'mg per dose': r['mg_per_dose'],
                 'Trough (mg/L)': round(r['last_cycle_min_mg_l'], 1),
                 'Peak (mg/L)': round(r['last_cycle_max_mg_l'], 1),
                 'Peak/trough': round(r['last_cycle_max_mg_l'] / r['last_cycle_min_mg_l'], 3),
                 'Average (mg/L)': round(r['last_cycle_mean_mg_l'], 1),
                 'Hours in range': round(r['within_h'], 1), 'Status': status(r)})
show('Table 7. Same daily total (6000 mg/day) split into more doses; trough, peak and average '
     'are for the final repeating cycle', rows, 'rq2_frequency_6000mg.csv')
theory_average = 6000 / (24 * base.ke_per_h * 4.60)
print(f'Pharmacokinetic theory [2]: average steady-state concentration = daily total / '
      f'(24 x excretion rate x volume) = {theory_average:.1f} mg/L for every frequency.')

rows = []
for daily in range(6000, 18001, 1000):
    r = lookup(grid, daily, 4, 4.60)
    rows.append({'Daily total (mg)': daily, 'mg per dose': r['mg_per_dose'],
                 'Trough (mg/L)': round(r['last_cycle_min_mg_l'], 1),
                 'Peak (mg/L)': round(r['last_cycle_max_mg_l'], 1),
                 'Peak/trough': round(r['last_cycle_max_mg_l'] / r['last_cycle_min_mg_l'], 3),
                 'Hours in range': round(r['within_h'], 1),
                 'First entry (h)': hours(r['first_entry_h']), 'Status': status(r)})
show('Table 8. Same frequency (4 doses/day), larger daily totals', rows, 'rq2_dosage_4_per_day.csv')

fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.4))
plot_concentration(left, [(f'{f} dose{"s" if f > 1 else ""}/day', simulate(Parameters(doses_per_day=f)))
                          for f in (1, 2, 3, 4, 6)], '6000 mg/day split into more doses (48 h)')
plot_concentration(right, [(f'{d:,} mg/day', simulate(Parameters(daily_mg=d, doses_per_day=4, duration_h=72.0)))
                           for d in (6000, 9000, 12000, 15000)], 'Larger daily totals at 4 doses/day (72 h)')
save_figure(fig, 'fig3_rq2_frequency_and_dosage.png')


# %% PART 5. RQ3 - which tested schedules stay within 800-1000 mg/L?
heading('PART 5. RQ3 - Which tested schedules stay within 800-1000 mg/L? (4.60 L)')
cells = {(r['daily_mg'], r['doses_per_day']): status(r) for r in grid if math.isclose(r['volume_l'], 4.60)}
show('Table 9. Final repeating cycle of every tested schedule at 4.60 L (168 h, dt = 0.01 h)',
     [{'Daily total (mg)': d, **{f'{f}/day': cells[d, f] for f in FREQUENCIES}} for d in DAILY_TOTALS],
     'rq3_status_map_4.60L.csv')
print('OK+ = in range from first entry to the end | OK = repeating cycle in range (dips below 800 '
      'during build-up)\nlow = never reaches 800 | dip = troughs below 800 | high = peaks at or '
      'above 1000 | swing = both limits crossed')

show('Table 10. Candidate schedules at 4.60 L, re-checked with dt = 0.005 h', [
    {'Daily total (mg)': r['daily_mg'], 'Doses/day': r['doses_per_day'], 'mg per dose': round(r['mg_per_dose'], 1),
     'First entry (h)': hours(r['first_entry_h']), 'Hours in range': round(r['within_h'], 1),
     'Trough (mg/L)': round(r['last_cycle_min_mg_l'], 1), 'Peak (mg/L)': round(r['last_cycle_max_mg_l'], 1),
     'Stays in range after first entry': yes_no(r['stays_in_range_after_first_entry']),
     'Repeating cycle in range': yes_no(r['last_cycle_in_range_and_converged']),
     'Same at dt 0.01': yes_no(r['same_result_at_dt_0.01'])}
    for r in refined if math.isclose(r['volume_l'], 4.60)], 'rq3_candidates_4.60L.csv')

# Selection: fewest tested daily doses, then lowest tested daily total.
criteria = (('Repeating cycle in range', 'last_cycle_in_range_and_converged'),
            ('Stays in range after first entry', 'stays_in_range_after_first_entry'))
selected = {}
for label, key in criteria:
    for volume in VOLUMES:
        passing = [r for r in refined if math.isclose(r['volume_l'], volume) and r[key]]
        selected[label, volume] = (min(passing, key=lambda r: (r['doses_per_day'], r['daily_mg']))
                                   if passing else None)
series = []
for label, _ in criteria:
    r = selected[label, 4.60]
    print(f'{label} at 4.60 L: ' + ('no tested schedule' if r is None else
          f"{r['daily_mg']:,} mg/day as {r['doses_per_day']} doses of {r['mg_per_dose']:,.0f} mg"))
    if r:
        series.append((f"{label}: {r['daily_mg']:,} mg/day, {r['doses_per_day']} doses/day",
                       simulate(Parameters(daily_mg=r['daily_mg'], doses_per_day=r['doses_per_day'],
                                           duration_h=EXTENDED_H))))
fig, ax = plt.subplots(figsize=(12, 4.4))
plot_concentration(ax, series, 'Selected tested schedules at 4.60 L (168 h)')
save_figure(fig, 'fig4_rq3_selected_4.60L.png')


# %% PART 6. RQ4 - blood volumes 3.68, 4.60 and 5.52 L (handout option 2)
heading('PART 6. RQ4 - Blood volumes 3.68 L (-20%), 4.60 L and 5.52 L (+20%)')


def passing_totals(volume, frequency, key='last_cycle_in_range_and_converged'):
    """Tested daily totals (re-checked at dt = 0.005 h) that meet a criterion."""
    return [r['daily_mg'] for r in refined if math.isclose(r['volume_l'], volume)
            and r['doses_per_day'] == frequency and r[key]]


def total_range(totals):
    return f'{min(totals):,} - {max(totals):,}' if totals else 'none'


rows = []
for volume in VOLUMES:
    for frequency in FREQUENCIES:
        cycle_ok = passing_totals(volume, frequency)
        strict = passing_totals(volume, frequency, 'stays_in_range_after_first_entry')
        rows.append({'Volume (L)': volume, 'Doses/day': frequency,
                     'Repeating cycle in range (mg/day)': total_range(cycle_ok),
                     'Tested totals': len(cycle_ok),
                     'Also in range after first entry (mg/day)': ', '.join(f'{d:,}' for d in strict) or 'none'})
show('Table 11. Tested daily totals that meet the limits, by blood volume', rows,
     'rq4_required_dosage_by_volume.csv')

show('Table 12. Selected schedule for each volume (fewest tested doses, then lowest tested total)', [
    {'Criterion': label, 'Volume (L)': volume,
     'Daily total (mg)': r['daily_mg'] if r else 'none', 'Doses/day': r['doses_per_day'] if r else '',
     'mg per dose': round(r['mg_per_dose'], 1) if r else '',
     'Trough (mg/L)': round(r['last_cycle_min_mg_l'], 1) if r else '',
     'Peak (mg/L)': round(r['last_cycle_max_mg_l'], 1) if r else '',
     'Hours in range': round(r['within_h'], 1) if r else ''}
    for (label, volume), r in selected.items()], 'rq4_selected_schedules.csv')

reference = selected['Repeating cycle in range', 4.60]
same_rows = [lookup(grid, reference['daily_mg'], reference['doses_per_day'], v) for v in VOLUMES]
show(f"Table 13. The 4.60 L schedule ({reference['daily_mg']:,} mg/day, "
     f"{reference['doses_per_day']} doses/day) given to all three volumes", [
    {'Volume (L)': r['volume_l'], 'Trough (mg/L)': round(r['last_cycle_min_mg_l'], 1),
     'Peak (mg/L)': round(r['last_cycle_max_mg_l'], 1), 'Hours in range': round(r['within_h'], 1),
     'Hours toxic': round(r['at_or_above_toxic_h'], 1), 'Status': status(r)} for r in same_rows],
     'rq4_same_schedule_three_volumes.csv')

fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.4))
plot_concentration(left, [(f'{v:.2f} L', simulate(Parameters(daily_mg=reference['daily_mg'],
                                                            doses_per_day=reference['doses_per_day'],
                                                            volume_l=v, duration_h=72.0)))
                          for v in VOLUMES],
                   f"{reference['daily_mg']:,} mg/day, {reference['doses_per_day']} doses/day "
                   'at three blood volumes (72 h)')
for offset, color, frequency in zip((-0.07, 0.0, 0.07), COLORS, (4, 6, 8)):
    for j, volume in enumerate(VOLUMES):
        cycle_ok = passing_totals(volume, frequency)
        if cycle_ok:
            right.plot([volume + offset] * 2, [min(cycle_ok), max(cycle_ok)], color=color, marker='o',
                       markersize=8, label=f'{frequency} doses/day' if j == 0 else None)
right.set(title='Tested daily totals that keep the repeating cycle in range',
          xlabel='Blood volume (L)', ylabel='Daily total (mg/day)', xticks=VOLUMES)
right.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
right.legend(loc='upper left')
save_figure(fig, 'fig5_rq4_blood_volume.png')


# %% PART 7. Supplementary: estimated dosage windows from the Euler runs
def estimated_dose_window(row):
    """Scale cycle extrema to the inclusive lower and exclusive upper limits."""
    lowest = row['daily_mg'] * row['medicinal_mg_l'] / row['last_cycle_min_mg_l']
    upper_limit = row['daily_mg'] * row['toxic_mg_l'] / row['last_cycle_max_mg_l']
    first, last = math.ceil(lowest), math.ceil(upper_limit) - 1
    return f'{first:,} - {last:,}' if first <= last else 'no whole-mg total'


heading('PART 7. Supplementary - estimated daily-total windows')
print(f'Scale the repeating-cycle trough and peak from Euler runs at dt = {base.dt_h:g} h.\n'
      'The trough must reach 800 mg/L and the peak must stay strictly below 1000 mg/L.\n'
      'The windows estimate whole-mg daily totals for the simulated repeating cycle.')
rows = []
for frequency in FREQUENCIES:
    row = {'Doses/day': frequency}
    for volume in VOLUMES:
        r = lookup(grid, 6000, frequency, volume)
        row['Peak/trough'] = round(r['last_cycle_max_mg_l'] / r['last_cycle_min_mg_l'], 3)
        row[f'{volume:.2f} L (mg/day)'] = estimated_dose_window(r)
    rows.append(row)
show(f'Table 14. Estimated daily totals for the repeating cycle (Euler dt = {base.dt_h:g} h)', rows,
     'supplementary_estimated_windows.csv')


# %% PART 8. Handout option 3 - dosage by body weight
heading('PART 8. Option 3 - Dosage by body weight (blood volume = 0.07 L per kg [4])')


def blood_l_per_kg(age_years=None):
    """Litres of blood per kg of body weight: more in infants and children than in adults [4]."""
    if age_years is None:
        return BLOOD_L_PER_KG
    return next(l_per_kg for under, l_per_kg in BLOOD_L_PER_KG_BY_AGE if age_years < under)


def blood_volume_l(weight_kg, age_years=None):
    """Estimate blood volume using the model's age-based litres per kg [4]."""
    return blood_l_per_kg(age_years) * weight_kg


def candidate_totals(p, step_mg):
    """Daily totals (multiples of step_mg) whose steady-state average is in range.

    The average is bioavailability x daily total / (24 x excretion rate x volume)
    (Part 4, [2]). A repeating cycle inside the limits has its average inside
    them too, so no other daily total can keep the cycle in range.
    """
    per_mg_l = 24 * p.ke_per_h * p.volume_l / p.bioavailability   # mg/day for an average of 1 mg/L
    return [k * step_mg for k in range(math.ceil(p.medicinal_mg_l * per_mg_l / step_mg),
                                       math.ceil(p.toxic_mg_l * per_mg_l / step_mg))]


def best_schedule(p, step_mg):
    """Fewest doses per day, then lowest daily total (the Part 5 rule), whose
    repeating cycle stays in range at dt = 0.01 h and again at REFINED_DT."""
    for frequency in FREQUENCIES:
        for daily in candidate_totals(p, step_mg):
            q = replace(p, daily_mg=daily, doses_per_day=frequency)
            if summarize(simulate(q))['last_cycle_in_range_and_converged']:
                fine = summarize(simulate(replace(q, dt_h=REFINED_DT)))
                if fine['last_cycle_in_range_and_converged']:
                    return fine
    return None


volume_weights = ', '.join(f'{v / BLOOD_L_PER_KG:.1f}' for v in VOLUMES)
print(f'Blood volume = {BLOOD_L_PER_KG} L/kg x body weight [4]: the Part 6 volumes 3.68, 4.60 and 5.52 L '
      f'belong to {volume_weights} kg.\n'
      'Only daily totals whose steady-state average is in range can work (Part 4 formula). Those are\n'
      f'tested in steps of {TOTAL_STEP} mg/day at every frequency; the fewest doses per day, then the lowest\n'
      'daily total, is chosen and re-checked with dt = 0.005 h (as in Parts 3 and 5).')
print('Check at 4.60 L: this search picks '
      f'{schedule_text(best_schedule(Parameters(duration_h=EXTENDED_H), TOTAL_STEP))}; Part 5 picked '
      f"{schedule_text(selected['Repeating cycle in range', 4.60])} from the full grid.")
chart = []
for weight in WEIGHTS:
    p = Parameters(volume_l=blood_volume_l(weight), duration_h=EXTENDED_H)
    chart.append((weight, p, best_schedule(p, TOTAL_STEP)))
show('Table 15. Dosing chart for the handout drug: chosen schedule by body weight (168 h)', [
    {'Weight (kg)': w, 'Blood volume (L)': round(p.volume_l, 2),
     'Daily totals tested (mg/day)': total_range(candidate_totals(p, TOTAL_STEP)),
     'Doses/day': r['doses_per_day'] if r else 'none', 'Daily total (mg)': r['daily_mg'] if r else '',
     'mg per dose': round(r['mg_per_dose'], 1) if r else '',
     'Trough (mg/L)': round(r['last_cycle_min_mg_l'], 1) if r else '',
     'Peak (mg/L)': round(r['last_cycle_max_mg_l'], 1) if r else '',
     'In range from first entry': yes_no(r['stays_in_range_after_first_entry']) if r else ''}
    for w, p, r in chart], 'option3_dosing_chart_by_weight.csv')


# %% PART 9. Handout option 4 - a real drug (published pharmacokinetics)
heading('PART 9. Option 4 - A real drug: published pharmacokinetics and validation')
# FILL IN with values from the drug's prescribing information (DailyMed or the FDA label,
# section "Clinical Pharmacology - Pharmacokinetics"); the handout suggests mesalamine
# (Apriso), Celebrex, Lotensin HCT and Synthroid [1]. Cite every value in the paper [5].
# Units: mg, hours, litres, kg. Concentrations in mg/L: 1 mcg/mL = 1 mg/L, 1 ng/mL = 0.001 mg/L.
REAL_DRUG = None
# REAL_DRUG = dict(
#     name='Drug name',
#     dose_mg=0,             # one dose of the usual adult regimen
#     doses_per_day=0,       # usual adult regimen: 1, 2, 3, 4, 6, 8, 12 or 24
#     half_life_h=0,         # elimination half-life
#     time_of_peak_h=0,      # Tmax after one dose; gives the absorption rate (or give ka_per_h=...)
#     bioavailability=0,     # fraction of a dose that reaches the blood, above 0 and at most 1
#     volume_l_per_kg=0,     # volume of distribution; if the label gives litres, divide by weight_kg;
#                            # if it gives Vd/F (apparent volume), use that with bioavailability=1
#     weight_kg=70,          # body weight of the label's study patients (70 if not stated)
#     medicinal_mg_l=0,      # lowest effective concentration (low end of the therapeutic range)
#     toxic_mg_l=0,          # concentration where toxic effects start (high end of the range)
#     daily_step_mg=0,       # smallest change of the daily total, e.g. one tablet strength
#     age_groups=[           # optional, other ages from the label ("Use in Specific Populations"):
#         (1, 12, 0, 0),     # (from age, up to but not including age, half-life h, volume L/kg)
#     ],
#     published={            # measured values to compare with; delete the ones you do not have
#         'Single-dose peak (mg/L)': 0,
#         'Steady-state peak (mg/L)': 0,
#         'Steady-state trough (mg/L)': 0,
#         'Steady-state average (mg/L)': 0,
#     },
# )
LOOK_UP = (  # (key, value to find, unit, where it is in the prescribing information)
    ('dose_mg', 'One dose of the usual adult regimen', 'mg', 'Dosage and Administration'),
    ('doses_per_day', 'Doses per day', 'per day', 'Dosage and Administration'),
    ('half_life_h', 'Elimination half-life', 'hours', 'Pharmacokinetics - Elimination'),
    ('time_of_peak_h', 'Time of peak concentration after one dose (Tmax)', 'hours', 'Pharmacokinetics - Absorption'),
    ('bioavailability', 'Bioavailability (fraction reaching the blood)', '0 to 1', 'Pharmacokinetics - Absorption'),
    ('volume_l_per_kg', 'Volume of distribution', 'L/kg', 'Pharmacokinetics - Distribution'),
    ('weight_kg', 'Body weight of the study patients', 'kg', 'Pharmacokinetics or Clinical Studies'),
    ('medicinal_mg_l', 'Lowest effective (therapeutic) concentration', 'mg/L', 'label or a pharmacology reference'),
    ('toxic_mg_l', 'Concentration where toxic effects start', 'mg/L', 'Overdosage, or a pharmacology reference'),
    ('daily_step_mg', 'Smallest change of the daily total (one tablet strength)', 'mg', 'How Supplied'),
    ('age_groups', 'Optional: half-life and volume for children or older adults', 'hours, L/kg',
     'Use in Specific Populations'),
    ('published', 'Measured concentrations, for validation', 'mg/L', 'Pharmacokinetics tables, or a study'))


def drug_parameters(drug, weight_kg, age_years=None):
    """The real drug's published rates replace the handout's [1]; the volume is its volume of
    distribution. A matching row of age_groups replaces the adult half-life and volume per kg.
    The run lasts 12 half-lives of the slower of absorption and excretion (at least 168 h),
    so the last cycle repeats the one before it."""
    half_life_h, l_per_kg = drug['half_life_h'], drug['volume_l_per_kg']
    for first_age, end_age, group_half_life_h, group_l_per_kg in drug.get('age_groups', ()):
        if age_years is not None and first_age <= age_years < end_age:
            half_life_h, l_per_kg = group_half_life_h, group_l_per_kg
    ka = drug.get('ka_per_h') or ka_from_peak_time(drug['time_of_peak_h'], half_life_h)
    slower_half_life_h = math.log(2.0) / min(ka, math.log(2.0) / half_life_h)
    return Parameters(daily_mg=drug['dose_mg'] * drug['doses_per_day'], doses_per_day=drug['doses_per_day'],
                      volume_l=l_per_kg * weight_kg, ka_per_h=ka,
                      half_life_h=half_life_h, bioavailability=drug['bioavailability'],
                      medicinal_mg_l=drug['medicinal_mg_l'], toxic_mg_l=drug['toxic_mg_l'],
                      duration_h=max(EXTENDED_H, 24.0 * math.ceil(12 * slower_half_life_h / 24)))


def predictions(p, dose_mg):
    """Model values a label or study can report: one dose (24 h run) and the usual regimen."""
    single = simulate(replace(p, daily_mg=dose_mg, doses_per_day=1, duration_h=24.0))
    c = single['concentration']
    peak = max(range(len(c)), key=c.__getitem__)
    usual = summarize(simulate(p))
    return single, usual, {
        'Single-dose peak (mg/L)': c[peak], 'Single-dose time of peak (h)': single['time'][peak],
        'Steady-state peak (mg/L)': usual['last_cycle_max_mg_l'],
        'Steady-state trough (mg/L)': usual['last_cycle_min_mg_l'],
        'Steady-state average (mg/L)': usual['last_cycle_mean_mg_l']}


drug_p = None
if REAL_DRUG is None:
    show('Table 16. Values to look up for option 4 (then fill in REAL_DRUG at the start of PART 9)',
         [{'Value': what, 'Unit': unit, 'Where in the prescribing information': where}
          for _, what, unit, where in LOOK_UP], 'option4_values_to_look_up.csv')
    print('Custom REAL_DRUG regimen skipped. The published Theoph dataset is compared in PART 9b.')
else:
    drug, name = REAL_DRUG, REAL_DRUG.get('name') or 'Real drug'
    missing = [key for key, *_ in LOOK_UP if key not in ('published', 'age_groups') and not drug.get(key)
               and not (key == 'time_of_peak_h' and drug.get('ka_per_h'))]
    if missing:
        raise ValueError('Fill in REAL_DRUG first: ' + ', '.join(missing))
    drug_p = drug_parameters(drug, drug['weight_kg'])
    single, usual, predicted = predictions(drug_p, drug['dose_mg'])
    show(f'Table 16. {name}: model parameters from the prescribing information [5]', [
        {'Parameter': 'Dose x doses per day', 'Value': f"{drug['dose_mg']:g} x {drug['doses_per_day']}",
         'Unit': 'mg', 'Source': 'label'},
        {'Parameter': 'Half-life', 'Value': drug['half_life_h'], 'Unit': 'hours', 'Source': 'label'},
        {'Parameter': 'Excretion rate = ln(2) / half-life', 'Value': sig(drug_p.ke_per_h), 'Unit': 'per hour',
         'Source': 'derived [2]'},
        {'Parameter': 'Absorption rate', 'Value': sig(drug_p.ka_per_h), 'Unit': 'per hour',
         'Source': 'label' if drug.get('ka_per_h') else f"derived from Tmax = {drug['time_of_peak_h']:g} h"},
        {'Parameter': 'Bioavailability', 'Value': drug_p.bioavailability, 'Unit': 'fraction', 'Source': 'label'},
        {'Parameter': 'Volume of distribution',
         'Value': f"{drug['volume_l_per_kg']:g} L/kg x {drug['weight_kg']:g} kg = {drug_p.volume_l:.2f}",
         'Unit': 'L', 'Source': 'label'},
        {'Parameter': 'Medicinal / toxic level', 'Value': f'{drug_p.medicinal_mg_l:g} / {drug_p.toxic_mg_l:g}',
         'Unit': 'mg/L', 'Source': 'label or reference'},
        {'Parameter': 'Run length / time step', 'Value': f'{drug_p.duration_h:g} / {drug_p.dt_h}',
         'Unit': 'hours', 'Source': 'study choice'},
    ], 'option4_real_drug_parameters.csv')

    theory = drug_p.bioavailability * drug_p.daily_mg / (24 * drug_p.ke_per_h * drug_p.volume_l)
    show(f"Table 17. {name}: model prediction for the usual regimen at {drug['weight_kg']:g} kg", [
        *({'Quantity': key, 'Model': sig(value)} for key, value in predicted.items()),
        {'Quantity': 'Steady-state average from theory [2] (mg/L)', 'Model': sig(theory)},
        {'Quantity': 'Repeating cycle in range / status',
         'Model': f"{yes_no(usual['last_cycle_in_range_and_converged'])} / {status(usual)}"},
        {'Quantity': f'Hours in range (of {drug_p.duration_h:g})', 'Model': round(usual['within_h'], 1)},
        {'Quantity': 'First entry into range (h)', 'Model': hours(usual['first_entry_h'])},
    ], 'option4_real_drug_prediction.csv')

    published = {key: value for key, value in drug.get('published', {}).items() if value}
    unknown = [key for key in published if key not in predicted]
    if unknown:
        raise ValueError(f'Unknown published values {unknown}; use these names: ' + ', '.join(predicted))
    validation = [{'Quantity': key, 'Published': value, 'Model': sig(predicted[key]),
                   'Difference (%)': round(100 * (predicted[key] - value) / value, 1)}
                  for key, value in published.items()]
    show(f'Table 18. Validation: model vs published values for {name}',
         validation or [{'Result': 'no published values in REAL_DRUG'}], 'option4_validation.csv')

    rows, series = [], []
    for factor in (0.8, 1.0, 1.2):
        q = drug_parameters(drug, drug['weight_kg'] * factor)
        run = simulate(q)
        r = summarize(run)
        rows.append({'Weight (kg)': round(drug['weight_kg'] * factor, 1), 'Volume (L)': round(q.volume_l, 2),
                     'Trough (mg/L)': sig(r['last_cycle_min_mg_l']), 'Peak (mg/L)': sig(r['last_cycle_max_mg_l']),
                     'Usual regimen': status(r),
                     'Best tested schedule (Part 8 search)': schedule_text(best_schedule(q, drug['daily_step_mg']))})
        series.append((f"{drug['weight_kg'] * factor:.0f} kg", run))
    show(f'Table 19. {name}: the usual regimen for 20% lighter and heavier patients', rows,
         'option4_body_weight.csv')

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.4))
    plot_concentration(left, [('Model, one dose', single)], f"{name}: one {drug['dose_mg']:g} mg dose")
    if 'Single-dose peak (mg/L)' in published:
        value = published['Single-dose peak (mg/L)']
        left.plot([drug.get('time_of_peak_h') or predicted['Single-dose time of peak (h)']], [value], 'x',
                  color=INK, markersize=10, mew=2, label='Published peak')
        left.set_ylim(0, max(left.get_ylim()[1], value * 1.1))
        left.legend(loc='best')
    plot_concentration(right, series, f'{name}: usual regimen at three body weights')
    save_figure(fig, 'fig6_real_drug.png')


# %% PART 9b. External comparison with the original, measured Theoph dataset
def compare_theoph_dataset(measured, provenance):
    """Predict an external single-dose study using fixed label parameters.

    No rate constant or volume is fitted to Theoph. Each subject's measured
    predose concentration is treated as an observed initial condition and
    decays with the same elimination constant. The untouched measurements,
    label assumptions, numerical error and prediction errors stay separate.
    """
    heading('PART 9b. Theophylline: model versus 132 real serum measurements')
    label_url = ('https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?'
                 'setid=1b685384-5d96-46fa-b93b-e96538a2cfe7')
    half_life, volume_per_kg, peak_time = 8.7, 0.45, 1.5
    ka = ka_from_peak_time(peak_time, half_life)
    comparison_dt, comparison_fine_dt = .001, .0005
    parameter_rows = [
        {'Parameter': 'Half-life', 'Value': half_life, 'Unit': 'h',
         'Basis': 'Label Table I: mean for otherwise healthy nonsmoking asthmatic adults; Theoph population details are unavailable.'},
        {'Parameter': 'Distribution volume per kg', 'Value': volume_per_kg, 'Unit': 'L/kg',
         'Basis': 'Label Distribution: ideal body weight. Observed weight is used as a proxy because ideal weight is unavailable.'},
        {'Parameter': 'Time to peak', 'Value': peak_time, 'Unit': 'h',
         'Basis': 'Assumed midpoint of the label interval 1-2 h, not a measured Theoph parameter.'},
        {'Parameter': 'Absorption rate', 'Value': ka, 'Unit': '1/h',
         'Basis': 'Derived from the assumed peak time and label half-life; not fitted to Theoph.'},
        {'Parameter': 'Bioavailability', 'Value': 1., 'Unit': 'fraction',
         'Basis': 'Model approximation to label complete absorption and no appreciable first-pass loss.'},
        {'Parameter': 'Numerical time step', 'Value': comparison_dt, 'Unit': 'h',
         'Basis': 'Smaller Euler step for faster theophylline absorption; checked again at 0.0005 h. Handout experiments keep 0.01 h.'},
        {'Parameter': 'Dose for each subject', 'Value': 'Dose x Wt', 'Unit': 'mg',
         'Basis': 'Use each recorded mg/kg dose and weight; give it only once.'},
        {'Parameter': 'Initial serum concentration', 'Value': 'Measured at Time = 0', 'Unit': 'mg/L',
         'Basis': 'Observed initial condition, decayed as C0 exp(-ke t); baseline is not fitted.'},
    ]
    show('Table 16a. Independent parameters and explicit assumptions for Theoph',
         parameter_rows, 'theoph_parameters.csv')
    points, subject_rows = [], []
    worst_refinement, worst_exact, plot_top = 0., 0., 0.
    fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharex=True, sharey=True)
    for ax, (subject, group) in zip(axes.flat, measured.groupby('Subject', sort=True)):
        group = group.sort_values('Time')
        if group['Wt'].nunique() != 1 or group['Dose'].nunique() != 1 or group.iloc[0]['Time'] != 0:
            raise ValueError(f'Inconsistent dose, weight or initial observation for subject {subject}.')
        weight, dose_per_kg = float(group.iloc[0]['Wt']), float(group.iloc[0]['Dose'])
        dose, c0 = weight * dose_per_kg, float(group.iloc[0]['conc'])
        p = Parameters(daily_mg=dose, doses_per_day=1, volume_l=volume_per_kg * weight,
                       ka_per_h=ka, half_life_h=half_life, duration_h=25., single_dose=True, dt_h=comparison_dt,
                       medicinal_mg_l=10., toxic_mg_l=20.)
        run = simulate(p)
        fine = simulate(replace(p, dt_h=comparison_fine_dt))
        if sum(value > 0 for value in run['intake']) != 1:
            raise AssertionError('A single-dose study received an extra dose.')
        estimates, errors = [], []
        for _, row in group.iterrows():
            t = float(row['Time'])
            index = round(t / p.dt_h)
            fine_index = round(t / comparison_fine_dt)
            initial = c0 * math.exp(-p.ke_per_h * t)
            estimate = run['concentration'][index] + initial
            refined = fine['concentration'][fine_index] + initial
            exact = exact_concentration(p, [t])[0] + initial
            observed = float(row['conc'])
            worst_refinement = max(worst_refinement, abs(estimate - refined))
            worst_exact = max(worst_exact, abs(estimate - exact))
            estimates.append(estimate)
            errors.append(estimate - observed)
            points.append({'Subject': int(subject), 'Weight_kg': weight, 'Dose_mg_per_kg': dose_per_kg,
                           'Single_dose_mg': dose, 'Time_h': t, 'Observed_mg_L': observed,
                           'Predicted_mg_L': estimate, 'Residual_predicted_minus_observed_mg_L': estimate - observed,
                           'Predicted_dt_0_0005_mg_L': refined, 'Initial_concentration_mg_L': c0})
        post_errors = errors[1:]
        subject_rows.append({'Subject': int(subject), 'Weight (kg)': weight, 'Single dose (mg)': dose,
                             'Post-dose samples': len(post_errors),
                             'Post-dose MAE (mg/L)': sum(abs(e) for e in post_errors) / len(post_errors),
                             'Post-dose RMSE (mg/L)': math.sqrt(sum(e * e for e in post_errors) / len(post_errors)),
                             'Observed sampled peak (mg/L)': float(group['conc'].max()),
                             'Predicted sampled peak (mg/L)': max(estimates)})
        curve = [c + c0 * math.exp(-p.ke_per_h * t) for t, c in zip(run['time'], run['concentration'])]
        plot_top = max(plot_top, max(curve), float(group['conc'].max()))
        ax.plot(run['time'], curve, color=COLORS[0], label='Fixed label model')
        ax.scatter(group['Time'], group['conc'], s=20, color=COLORS[1], zorder=3, label='Measured serum')
        ax.set_title(f'Subject {subject} | {weight:g} kg', fontsize=10)
        ax.set_xlim(0, 25)
    for ax in axes.flat:
        ax.set_ylim(0, plot_top * 1.08)
    for ax in axes[-1, :]:
        ax.set_xlabel('Time after one dose (h)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Concentration (mg/L)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5, .965))
    fig.suptitle('Theophylline: real observations versus predictions without fitting', y=.995)
    save_figure(fig, 'fig6_theoph_observed_vs_model.png', layout_rect=(0, 0, 1, .93))
    frame = pd.DataFrame(points)
    frame.to_csv(os.path.join(OUT, 'theoph_observed_vs_predicted.csv'), index=False)
    show('Table 17a. Theoph prediction error by subject (10 post-dose samples each)',
         [{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()} for row in subject_rows],
         'theoph_subject_metrics.csv')
    metrics = []
    # Predose samples are used as initial conditions. The primary error score
    # therefore excludes all 12 predose samples rather than rewarding the model for them.
    for label, subset in [('Post-dose observations (primary)', frame[frame['Time_h'] > 0]),
                          ('All observations (includes supplied initial conditions)', frame)]:
        error = subset['Residual_predicted_minus_observed_mg_L']
        residual_ss = float((error ** 2).sum())
        total_ss = float(((subset['Observed_mg_L'] - subset['Observed_mg_L'].mean()) ** 2).sum())
        metrics.append({'Evaluation': label, 'N': len(subset),
                        'MAE (mg/L)': float(error.abs().mean()),
                        'RMSE (mg/L)': math.sqrt(residual_ss / len(subset)),
                        'Mean bias (mg/L)': float(error.mean()),
                        'R squared (descriptive)': 1 - residual_ss / total_ss})
    show('Table 18a. External prediction error; no parameters fitted to Theoph',
         [{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()} for row in metrics],
         'theoph_overall_metrics.csv')
    if worst_exact > .01 or worst_refinement > .01:
        raise AssertionError(f'Numerical error is too large: exact {worst_exact:g}, refinement {worst_refinement:g} mg/L.')
    report = {'dataset': provenance, 'label_url': label_url, 'label_revision': '2026-02-06',
              'parameters': parameter_rows, 'metrics': metrics,
              'max_sampled_difference_dt_0_001_vs_0_0005_mg_L': worst_refinement,
              'max_sampled_euler_error_vs_exact_mg_L': worst_exact,
              'parameter_fitting': 'None; all 12 subjects evaluated with the same fixed label parameters.',
              'interpretation': 'External comparison with real data, not proof of clinical validity or individualized dosage accuracy.',
              'limitations': ['Individual age, smoking status and ideal body weight are not supplied.',
                              'The dataset documentation does not identify the oral formulation; the reference label describes immediate-release absorption.',
                              'Adult population mean half-life is an approximation for these subjects.',
                              'Tmax = 1.5 h is an assumed midpoint of a published interval.',
                              'Observed weight substitutes for the label ideal body weight.',
                              'Single-dose measurements do not validate repeating-dose schedules or pediatric predictions.',
                              'The source records serum concentrations; no serum/plasma conversion factor is estimated.']}
    Path(OUT, 'theoph_validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    post = frame[frame['Time_h'] > 0]
    top = max(float(post['Observed_mg_L'].max()), float(post['Predicted_mg_L'].max())) * 1.05
    left.scatter(post['Observed_mg_L'], post['Predicted_mg_L'], s=24, alpha=.7, color=COLORS[0])
    left.plot([0, top], [0, top], '--', color=INK, label='Perfect agreement')
    left.set(xlabel='Measured serum concentration (mg/L)', ylabel='Predicted concentration (mg/L)',
             title='120 post-dose samples', xlim=(0, top), ylim=(0, top))
    left.legend()
    right.scatter(post['Time_h'], post['Residual_predicted_minus_observed_mg_L'], s=24, alpha=.7, color=COLORS[1])
    right.axhline(0, color=INK, ls='--')
    right.set(xlabel='Time after one dose (h)', ylabel='Prediction minus measurement (mg/L)',
              title='Errors remain visible')
    save_figure(fig, 'fig8_theoph_prediction_errors.png')
    print(f'Numerical check: max change after halving dt = {worst_refinement:.4f} mg/L; '
          f'max Euler error versus exact = {worst_exact:.4f} mg/L.')
    print('Theoph: original observations retained; no clinical pass/fail threshold assumed. '
          'The calculator continues to use the hypothetical handout drug.')
    return report


theoph_comparison = compare_theoph_dataset(theoph_data, theoph_provenance)


# %% PART 10. Summary of findings (every number below is computed above)
heading('SUMMARY OF FINDINGS')
FINDINGS = []       # the sentences below, also shown on the web application (PART 11)


def finding(text):
    FINDINGS.append(text)
    print(text)


by_frequency = {f: lookup(grid, 6000, f, 4.60) for f in FREQUENCIES}
ratio = {f: r['last_cycle_max_mg_l'] / r['last_cycle_min_mg_l'] for f, r in by_frequency.items()}
strict_pick = selected['Stays in range after first entry', 4.60]
three_window = estimated_dose_window(by_frequency[3])
volume_ranges = '; '.join(f"{v:.2f} L: {total_range(passing_totals(v, reference['doses_per_day']))} mg/day"
                          for v in VOLUMES)
finding(f"RQ1  6000 mg once a day peaks at {baseline['concentration'][peak_index]:.1f} mg/L "
      f"({baseline['time'][peak_index]:.2f} h) "
      f"and never reaches 800 mg/L in 48 h. After a week it repeats between "
      f"{base168['last_cycle_min_mg_l']:.1f} and {base168['last_cycle_max_mg_l']:.1f} mg/L: "
      'the baseline dosage is too low.')
finding(f"RQ2  Splitting 6000 mg/day into more doses keeps the average near {theory_average:.0f} mg/L "
      f"but narrows the swing (peak/trough {ratio[1]:.2f} at 1 dose/day, {ratio[6]:.2f} at 6 doses/day). "
      'A larger daily total multiplies trough and peak by the same factor.')
finding(f"RQ3  The limits allow peak/trough < 1.25, which the 1- and 2-dose cycles exceed. "
      f"At 4.60 L, the estimated Euler window for 3 doses/day is {three_window} mg/day (Table 14), "
      f"too narrow for the tested grid. "
      f"At 4.60 L the fewest tested doses with the repeating cycle in range: {reference['daily_mg']:,} "
      f"mg/day as {reference['doses_per_day']} doses/day. In range from first entry onward: "
      + ('none' if strict_pick is None else
         f"{strict_pick['daily_mg']:,} mg/day as {strict_pick['doses_per_day']} doses/day") + '.')
finding(f"RQ4  At {reference['doses_per_day']} doses/day the tested totals that work grow with blood "
      f"volume ({volume_ranges}). The 4.60 L schedule is '{status(same_rows[0])}' at 3.68 L and "
      f"'{status(same_rows[2])}' at 5.52 L, so the dosage must follow blood volume.")
finding('Option 3  With blood volume = 0.07 L/kg x body weight [4], the chosen schedule (Table 15) grows with '
      'weight: ' + '; '.join(f'{w} kg: {schedule_text(r)}' for w, _, r in chart) + '.')
finding('RQ6  The single-dose Theoph comparison uses 120 post-dose observations. '
        f"Mean absolute error: {theoph_comparison['metrics'][0]['MAE (mg/L)']:.4f} mg/L (PART 9b).")
if drug_p is not None:
    finding(f"Option 4  {name}: the usual regimen repeats between {sig(usual['last_cycle_min_mg_l'])} and "
          f"{sig(usual['last_cycle_max_mg_l'])} mg/L (target {drug_p.medicinal_mg_l:g}-{drug_p.toxic_mg_l:g} "
          f"mg/L, status '{status(usual)}'). Model vs published: "
          + ('; '.join(f"{v['Quantity']} {v['Difference (%)']:+.1f}%" for v in validation) or 'none entered') + '.')
finding(f"Verification: {sum(row['Status'] == 'PASS' for row in checks)} of {len(checks)} checks passed. "
      + ('This verifies the code. Theoph prediction errors are reported separately in PART 9b.' if drug_p is None else
         f'This verifies the code; Table 18 compares the model with published measurements of {name}.'))


# %% PART 11. Web application: the study's calculations, and the dosage for a patient (handout option 3)
heading("PART 11. Web application - the study's calculations and a patient's dosage")
HANDOUT_NAME = 'Handout drug'
DRUGS = [HANDOUT_NAME] + ([name] if drug_p is not None else [])


def recommend(drug_name, age_years, weight_kg):
    """The Part 8 search for one patient, and the text the calculator shows about it."""
    if drug_name == HANDOUT_NAME:
        p, step = Parameters(volume_l=blood_volume_l(weight_kg, age_years), duration_h=EXTENDED_H), TOTAL_STEP
        volume = f'Blood volume {p.volume_l:.2f} L = {1000 * blood_l_per_kg(age_years):g} mL/kg x {weight_kg:g} kg'
        adult_values = True
    else:
        p, step = drug_parameters(REAL_DRUG, weight_kg, age_years), REAL_DRUG['daily_step_mg']
        volume = f'Volume of distribution {p.volume_l:.1f} L'
        adult_values = not any(first <= age_years < end for first, end, *_ in REAL_DRUG.get('age_groups', ()))
    for size in (step, step / 5, step / 50):      # finer dose steps when a small patient fits none
        best = best_schedule(p, size)
        if best is not None:
            break
    low, high = p.medicinal_mg_l, p.toxic_mg_l
    result = dict(p=p, best=best, run=None, title='', headline='No tested schedule fits',
                  details=[volume, f'No dose keeps the level between {low:g} and {high:g} mg/L.'])
    if best is None:
        return result
    n = best['doses_per_day']
    result['run'] = simulate(replace(p, daily_mg=best['daily_mg'], doses_per_day=n))
    result['title'] = f'{schedule_text(best)} ({weight_kg:g} kg, age {age_years:g})'
    result['headline'] = f"{sig(best['mg_per_dose']):,g} mg every {best['interval_h']:g} hours"
    result['details'] = [
        f"{n} dose{'s' if n > 1 else ''} a day = {best['daily_mg']:,g} mg per day", volume,
        f"After a week the level repeats between {sig(best['last_cycle_min_mg_l']):g} and "
        f"{sig(best['last_cycle_max_mg_l']):g} mg/L (target {low:g} to {high:g} mg/L).",
        f"It first reaches {low:g} mg/L after {hours(best['first_entry_h'])} h"
        + ('.' if best['stays_in_range_after_first_entry'] else ' and dips below it during the first days.')]
    if size < step:
        result['details'].append(f'Small patient: no dose fits in steps of {step:g} mg/day, so steps of '
                                 f'{size:g} mg/day were used.')
    if age_years < 18 and adult_values:
        result['details'].append(
            'Age affects estimated blood volume only. Absorption and half-life use the handout values.'
            if drug_name == HANDOUT_NAME else 'Under 18: the label values are for adults (add age_groups).')
    return result


def ask_number(question, default, low, high):
    """A number typed under the output. Enter, a bad answer or no keyboard (a
    notebook run in the background, a script run without a terminal) gives the default."""
    try:
        answer = input(f'{question}, {low:g} to {high:g} (press Enter for {default:g}): ').strip()
    except (EOFError, RuntimeError):
        print(f'No keyboard here; using {default:g}.')
        return default
    try:
        value = float(answer) if answer else default
    except ValueError:
        value = math.nan
    if not low <= value <= high:
        print(f'"{answer}" is not from {low:g} to {high:g}; using {default:g}.')
        return default
    return value


def typed_calculator():
    """WEB_APP = False: the age and weight are typed under the output instead of on the web page."""
    age = ask_number("Patient's age in years", 30, 0, 110)
    weight = ask_number("Patient's weight in kg", 70, 2, 250)
    results = [(drug_name, recommend(drug_name, age, weight)) for drug_name in DRUGS]
    show(f'Table 20. Recommended dosage for a patient aged {age:g} weighing {weight:g} kg', [
        {'Drug': drug_name, 'Dosage': r['headline'],
         'Per day (mg)': f"{r['best']['daily_mg']:,g}" if r['best'] else '',
         'Volume (L)': round(r['p'].volume_l, 2),
         'Trough (mg/L)': f"{sig(r['best']['last_cycle_min_mg_l']):g}" if r['best'] else '',
         'Peak (mg/L)': f"{sig(r['best']['last_cycle_max_mg_l']):g}" if r['best'] else '',
         'In range from first entry': yes_no(r['best']['stays_in_range_after_first_entry']) if r['best'] else ''}
        for drug_name, r in results], 'calculator_result.csv')
    for drug_name, r in results:
        print(f'\n{drug_name}:\n  ' + '\n  '.join(r['details']))
    panels = [(f'{drug_name}: {r["title"]}', r['run']) for drug_name, r in results if r['run']]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(max(10, 6.5 * len(panels)), 4.2), squeeze=False)
        for ax, (title, run) in zip(axes[0], panels):
            plot_concentration(ax, [(title, run)], title)
        save_figure(fig, 'fig7_patient_dose.png')


WEB_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drug Dosage Calculator</title>
<link rel="icon" href="data:,">
<style>
  :root { --ink: #2b2a27; --muted: #5f5e59; --line: #e1e0d9; --field: #c3c2b7; --page: #f6f5f1;
          --blue: #1d63b8; --red: #d03b3b; }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body { margin: 0; background: var(--page); color: var(--ink);
         font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  main { max-width: 1120px; margin: 0 auto; padding: 32px 16px 40px; }
  h1 { margin: 0; font-size: 28px; line-height: 1.2; }
  .lead { margin: 6px 0 0; color: var(--muted); }
  .tabs { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 20px; border-bottom: 1px solid var(--line); }
  .tabs button { padding: 9px 14px; border: 0; border-bottom: 3px solid transparent; border-radius: 8px 8px 0 0;
                 background: none; color: var(--muted); }
  .tabs button:hover { color: var(--ink); }
  .tabs button[aria-selected="true"] { background: #fff; border-bottom-color: var(--blue); color: var(--blue); }
  .intro { margin: 16px 0 0; color: var(--muted); }
  .top { display: grid; grid-template-columns: minmax(280px, 360px) minmax(0, 1fr); gap: 20px; margin-top: 16px; }
  .card { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 20px; min-width: 0; }
  .below { margin-top: 20px; }
  .gap { margin-top: 20px; }
  h2 { margin: 0 0 12px; font-size: 13px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
  label { display: block; font-weight: 600; margin: 12px 0 4px; }
  input, select { width: 100%; font: inherit; padding: 9px 11px; color: inherit; background: #fff;
                  border: 1px solid var(--field); border-radius: 8px; }
  input:focus, select:focus, button:focus-visible, summary:focus-visible {
    outline: 2px solid var(--blue); outline-offset: 1px; }
  button { font: inherit; font-weight: 600; border-radius: 8px; cursor: pointer; }
  .go { width: 100%; margin-top: 18px; padding: 11px; border: 0; background: var(--blue); color: #fff; }
  .go:disabled { opacity: .6; cursor: progress; }
  .quiet { padding: 8px 12px; border: 1px solid var(--field); background: #fff; color: var(--ink); text-align: left; }
  .presets { display: grid; gap: 6px; margin-bottom: 14px; }
  details { margin-top: 14px; }
  summary { cursor: pointer; font-weight: 600; }
  .result { margin-top: 24px; padding-top: 20px; border-top: 1px solid var(--line); }
  .headline { margin: 0; font-size: 24px; font-weight: 700; line-height: 1.25; color: var(--blue); }
  .headline.problem { color: var(--red); }
  .points { margin: 10px 0 0; padding-left: 18px; color: var(--muted); }
  .points li { margin: 4px 0; }
  img { display: block; width: 100%; height: auto; }
  .note { margin: 10px 0 0; font-size: 14px; color: var(--muted); }
  .wide { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { padding: 6px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
  th { font-weight: 600; color: var(--muted); background: #faf9f6; }
  .facts td:first-child { width: 45%; color: var(--muted); }
  #c-table td:first-child { white-space: nowrap; }
  .rows th, .rows td { padding: 3px; border: 0; background: none; }
  .rows input, .rows select { padding: 7px; font-size: 14px; }
  .map td, .map th { text-align: center; padding: 3px 6px; }
  .map td { border: 1px solid #fff; font-weight: 600; }
  .map td:first-child { font-weight: 400; white-space: nowrap; }
  .chip { display: inline-block; padding: 1px 8px; border-radius: 6px; font-weight: 600; font-size: 14px;
          letter-spacing: 0; text-transform: none; }
  .s-okplus { background: #bfe6bf; color: #0f3d0f; }
  .s-ok { background: #def2de; color: #145214; }
  .s-low { background: #eceae4; color: #4a4943; }
  .s-dip { background: #fdeccb; color: #6b4a00; }
  .s-high { background: #f8d5d2; color: #7a1712; }
  .s-swing { background: #f0c8e0; color: #6a1147; }
  .s-nc { background: #fff; color: var(--muted); }
  .graphs { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
  figure { margin: 0; }
  figure img { border: 1px solid var(--line); border-radius: 8px; }
  figcaption { margin-top: 4px; font-size: 13px; color: var(--muted); }
  footer { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
           margin-top: 20px; font-size: 14px; color: var(--muted); }
  #stop { padding: 8px 14px; border: 1px solid var(--field); background: #fff; color: var(--ink); }
  @media (max-width: 760px) { .top, .graphs { grid-template-columns: minmax(0, 1fr); } }
</style>
</head>
<body>
<main>
  <h1>Drug Dosage Calculator</h1>
  <p class="lead">Run the calculations from our drug dosage study. Every tab uses the same model as the rest of the program.</p>
  <nav class="tabs" id="tabs" role="tablist" aria-label="Calculations">
    <button role="tab" type="button" data-tab="patient">Patient dosage</button>
    <button role="tab" type="button" data-tab="schedule">Try a schedule</button>
    <button role="tab" type="button" data-tab="compare">Compare schedules</button>
    <button role="tab" type="button" data-tab="find">Find schedules</button>
    <button role="tab" type="button" data-tab="measured">Measured data</button>
    <button role="tab" type="button" data-tab="results">Study results</button>
  </nav>

  <section class="tab" id="tab-patient" role="tabpanel">
    <p class="intro"><strong>RQ5.</strong> Enter a patient's age and weight to find the dose that keeps the drug between
      its medicinal and toxic levels. The handout drug is hypothetical, so this is a model result, not a prescription.</p>
    <div class="top">
      <section class="card">
        <h2>Patient</h2>
        <form id="form-patient">
          <label for="age">Age (years)</label>
          <input id="age" name="age" type="number" min="0" max="110" step="any" value="30" required>
          <label for="weight">Weight (kg)</label>
          <input id="weight" name="weight" type="number" min="2" max="250" step="any" value="70" required>
          <label for="drug">Drug</label>
          <select id="drug" name="drug">__DRUGS__</select>
          <button class="go" type="submit">Calculate dosage</button>
        </form>
        <div class="result" aria-live="polite">
          <h2>Recommended dosage</h2>
          <p class="headline" id="p-headline">Calculating...</p>
          <ul class="points" id="p-details"></ul>
        </div>
      </section>
      <section class="card">
        <h2>Drug level over time</h2>
        <img id="p-graph" alt="Drug concentration over time for the recommended dosage" hidden>
      </section>
    </div>
  </section>

  <section class="tab" id="tab-schedule" role="tabpanel" hidden>
    <p class="intro"><strong>RQ1.</strong> Enter any schedule and see how the drug level behaves. The starting values
      are the handout's baseline: 6,000 mg once a day at 4.60 L for 48 hours.</p>
    <div class="top">
      <section class="card">
        <h2>Schedule</h2>
        <form id="form-schedule">
          <label for="s-daily">Daily dose (mg/day)</label>
          <input id="s-daily" name="daily" type="number" min="1" max="1000000" step="any" value="6000" required>
          <label for="s-doses">Doses per day</label>
          <select id="s-doses" name="doses">__DOSES__</select>
          <label for="s-volume">Blood volume (L)</label>
          <input id="s-volume" name="volume" type="number" min="0.05" max="200" step="any" value="4.60" required>
          <label for="s-hours">Run length</label>
          <select id="s-hours" name="hours">
            <option value="48">48 hours (handout baseline)</option>
            <option value="72">72 hours</option>
            <option value="168">1 week</option>
            <option value="336">2 weeks</option>
          </select>
          <details>
            <summary>Drug values (handout defaults)</summary>
            <label for="s-ka">Absorption rate (per hour)</label>
            <input id="s-ka" name="ka" type="number" min="0.01" max="20" step="any" value="0.25" required>
            <label for="s-half">Half-life (hours)</label>
            <input id="s-half" name="half_life" type="number" min="0.1" max="200" step="any" value="6" required>
            <label for="s-bio">Bioavailability (0 to 1)</label>
            <input id="s-bio" name="bioavailability" type="number" min="0.01" max="1" step="any" value="1" required>
            <label for="s-low">Medicinal level (mg/L)</label>
            <input id="s-low" name="low" type="number" min="0.001" max="100000" step="any" value="800" required>
            <label for="s-high">Toxic level (mg/L)</label>
            <input id="s-high" name="high" type="number" min="0.001" max="1000000" step="any" value="1000" required>
          </details>
          <button class="go" type="submit">Run schedule</button>
        </form>
      </section>
      <section class="card">
        <div aria-live="polite">
          <h2>Result <span class="chip" id="s-status"></span></h2>
          <p class="headline" id="s-headline">Calculating...</p>
          <ul class="points" id="s-details"></ul>
        </div>
        <h2 class="gap">Drug level and drug amounts over time</h2>
        <img id="s-graph" alt="Drug concentration and the amount in each part of the model over time" hidden>
      </section>
    </div>
    <section class="card below">
      <h2>The numbers</h2>
      <table class="facts" id="s-facts"></table>
      <details>
        <summary id="s-values-title">Values over time</summary>
        <div class="wide"><table id="s-values"></table></div>
      </details>
    </section>
  </section>

  <section class="tab" id="tab-compare" role="tabpanel" hidden>
    <p class="intro"><strong>RQ2 and RQ4.</strong> Put up to five schedules side by side. Each one runs for a week,
      like the study's tables. Leave a row's daily dose empty to skip it.</p>
    <div class="top">
      <section class="card">
        <h2>Start from an experiment in the study</h2>
        <div class="presets">
          <button class="quiet" type="button" data-preset="split">Same daily total, more doses</button>
          <button class="quiet" type="button" data-preset="totals">Larger daily totals</button>
          <button class="quiet" type="button" data-preset="volumes">Same schedule, three blood volumes</button>
        </div>
        <h2>Schedules</h2>
        <form id="form-compare">
          <table class="rows">
            <thead><tr><th></th><th>mg/day</th><th>Doses/day</th><th>Volume (L)</th></tr></thead>
            <tbody id="c-rows"></tbody>
          </table>
          <label for="c-span">Graph shows</label>
          <select id="c-span" name="span">
            <option value="48">First 48 hours</option>
            <option value="72">First 72 hours</option>
            <option value="168">The whole week</option>
          </select>
          <button class="go" type="submit">Compare</button>
        </form>
      </section>
      <section class="card">
        <h2>Drug level over time</h2>
        <p class="headline" id="c-headline">Calculating...</p>
        <ul class="points" id="c-details"></ul>
        <img id="c-graph" alt="Drug concentration over time for each schedule" hidden>
      </section>
    </div>
    <section class="card below">
      <h2>Last repeating cycle of each schedule</h2>
      <div class="wide"><table id="c-table"></table></div>
      <p class="note legend"></p>
    </section>
  </section>

  <section class="tab" id="tab-find" role="tabpanel" hidden>
    <p class="intro"><strong>RQ3 and RQ4.</strong> Test daily totals and dosing frequencies at one blood volume.</p>
    <div class="top">
      <section class="card">
        <h2>Blood volume</h2>
        <form id="form-find">
          <label for="f-volume">Blood volume (L)</label>
          <input id="f-volume" name="volume" type="number" min="0.05" max="200" step="any" value="4.60" required>
          <details>
            <summary>Daily totals to test</summary>
            <label for="f-first">Lowest (mg/day)</label>
            <input id="f-first" name="first" type="number" min="1" max="1000000" step="any" value="6000" required>
            <label for="f-last">Highest (mg/day)</label>
            <input id="f-last" name="last" type="number" min="1" max="1000000" step="any" value="18000" required>
            <label for="f-step">Step (mg/day)</label>
            <input id="f-step" name="step" type="number" min="1" max="1000000" step="any" value="500" required>
          </details>
          <button class="go" type="submit">Test all schedules</button>
        </form>
      </section>
      <section class="card">
        <div aria-live="polite">
          <h2>Fewest doses that work</h2>
          <p class="headline" id="f-headline">Calculating...</p>
          <ul class="points" id="f-details"></ul>
        </div>
        <h2 class="gap">Selected schedules over one week</h2>
        <img id="f-graph" alt="Drug concentration over one week for the selected schedules" hidden>
      </section>
    </div>
    <section class="card below">
      <h2>Status at the initial step of 0.01 h</h2>
      <div class="wide"><table class="map" id="f-map"></table></div>
      <p class="note legend"></p>
      <h2 class="gap">Status changes at 0.005 h</h2>
      <p class="note" id="f-refinement-note"></p>
      <div class="wide"><table id="f-refinement"></table></div>
    </section>
    <section class="card below">
      <h2>Estimated daily totals for the repeating cycle</h2>
      <div class="wide"><table id="f-windows"></table></div>
      <p class="note">Whole-mg estimates scale the Euler trough and peak from the 0.01 h run. The lower concentration
        limit is inclusive and the upper limit is exclusive.</p>
    </section>
  </section>

  <section class="tab" id="tab-measured" role="tabpanel" hidden>
    <p class="intro"><strong>RQ6.</strong> Compare the model with real theophylline measurements from 12 people after one
      dose. The model uses fixed published values, with nothing fitted to these measurements.</p>
    <div class="top">
      <section class="card">
        <h2>Person</h2>
        <form id="form-measured">
          <label for="m-subject">Person in the dataset</label>
          <select id="m-subject" name="subject">__SUBJECTS__</select>
          <button class="go" type="submit">Compare with the model</button>
        </form>
        <div class="result" aria-live="polite">
          <h2>This person</h2>
          <p class="headline" id="m-headline">Calculating...</p>
          <ul class="points" id="m-details"></ul>
        </div>
      </section>
      <section class="card">
        <h2>Measured points and model curve</h2>
        <img id="m-graph" alt="Measured concentrations and the model curve for this person" hidden>
      </section>
    </div>
    <section class="card below">
      <h2>All 12 people, 120 samples after the dose</h2>
      <table class="facts" id="m-overall"></table>
      <details>
        <summary>This person's samples</summary>
        <div class="wide"><table id="m-samples"></table></div>
      </details>
      <details>
        <summary>Errors for each person</summary>
        <div class="wide"><table id="m-people"></table></div>
      </details>
    </section>
  </section>

  <section class="tab" id="tab-results" role="tabpanel" hidden>
    <section class="card below">
      <h2>Summary of findings</h2>
      <ul class="points">__FINDINGS__</ul>
    </section>
    <section class="card below">
      <h2>Verification checks</h2>
      <div class="wide"><table><thead><tr><th>Check</th><th>Result</th><th>Status</th></tr></thead>
        <tbody>__CHECKS__</tbody></table></div>
    </section>
    <section class="card below">
      <h2>Study graphs (saved in the results folder)</h2>
      <div class="graphs">__GRAPHS__</div>
    </section>
  </section>

  <footer>
    <span>Balagao, Insigne, Osorio. A school project model, not medical advice.</span>
    <button id="stop" type="button">Stop calculator</button>
  </footer>
</main>
<script>
const $ = id => document.getElementById(id);
const STATUS = {'OK+': 's-okplus', OK: 's-ok', low: 's-low', dip: 's-dip', high: 's-high', swing: 's-swing', 'n/c': 's-nc'};
const REFERENCE = __REFERENCE__;
const PRESETS = {   // the study's experiments: graph length, then [mg/day, doses/day, volume]
  split: [48, [1, 2, 3, 4, 6].map(doses => [6000, doses, 4.6])],
  totals: [72, [6000, 9000, 12000, 15000].map(daily => [daily, 4, 4.6])],
  volumes: [72, [3.68, 4.6, 5.52].map(volume => [REFERENCE[0], REFERENCE[1], volume])],
};
const LEGEND = [['OK+', 'in range from first entry to the end'], ['OK', 'repeating cycle in range, dips during build-up'],
  ['low', 'never reaches the medicinal level'], ['dip', 'troughs fall below it'], ['high', 'peaks reach the toxic level'],
  ['swing', 'both limits crossed'], ['n/c', 'not settled yet']];

async function ask(path, query) {
  try {
    return await (await fetch(path + '?' + new URLSearchParams(query))).json();
  } catch (error) {
    return {found: false, headline: 'The calculator has stopped', details: ['Run the Python file again to restart it.']};
  }
}

function fillList(list, items) {
  list.replaceChildren(...items.map(text => {
    const item = document.createElement('li');
    item.textContent = text;
    return item;
  }));
}

function fillTable(table, headers, rows, statusFrom = Infinity) {
  table.replaceChildren();
  if (headers.length) {
    const head = table.createTHead().insertRow();
    headers.forEach(text => {
      const cell = document.createElement('th');
      cell.scope = 'col';
      cell.textContent = text;
      head.append(cell);
    });
  }
  const body = table.createTBody();
  rows.forEach(values => {
    const row = body.insertRow();
    values.forEach((text, index) => {
      const cell = row.insertCell();
      cell.textContent = text;
      if (index >= statusFrom) cell.className = STATUS[text] || '';
    });
  });
}

function showAnswer(prefix, answer) {
  const headline = $(prefix + '-headline');
  headline.textContent = answer.headline;
  headline.classList.toggle('problem', !answer.found);
  fillList($(prefix + '-details'), answer.details || []);
  const graph = $(prefix + '-graph');
  graph.hidden = !answer.graph;
  if (answer.graph) graph.src = 'data:image/png;base64,' + answer.graph;
}

function connect(formId, path, show) {    // a form that sends its fields to the program and shows the answer
  const form = $(formId), button = form.querySelector('button[type=submit]'), label = button.textContent;
  let latestRun = 0;
  const run = async event => {
    if (event) event.preventDefault();
    const runId = ++latestRun;
    button.disabled = true;
    button.textContent = 'Calculating...';
    const answer = await ask(path, Object.fromEntries(new FormData(form)));
    if (runId !== latestRun) return;
    show(answer);
    button.disabled = false;
    button.textContent = label;
  };
  form.addEventListener('submit', run);
  return run;
}

const runPatient = connect('form-patient', '/api/dose', answer => showAnswer('p', answer));

const runSchedule = connect('form-schedule', '/api/schedule', answer => {
  showAnswer('s', answer);
  const chip = $('s-status');
  chip.textContent = answer.status || '';
  chip.className = 'chip ' + (STATUS[answer.status] || '');
  chip.hidden = !answer.status;
  fillTable($('s-facts'), [], answer.facts || []);
  $('s-values-title').textContent = answer.values_title || 'Values over time';
  fillTable($('s-values'), answer.value_columns || [], answer.value_rows || []);
});

const compareForm = $('form-compare');
for (let i = 0; i < 5; i++) {
  const letter = 'ABCDE'[i], row = document.createElement('tr');
  row.innerHTML = `<td>${letter}</td>
    <td><input name="daily${i}" type="number" min="1" max="1000000" step="any" aria-label="Schedule ${letter} daily dose"></td>
    <td><select name="doses${i}" aria-label="Schedule ${letter} doses per day">__DOSES__</select></td>
    <td><input name="volume${i}" type="number" min="0.05" max="200" step="any" aria-label="Schedule ${letter} blood volume"></td>`;
  $('c-rows').append(row);
}
const runCompare = connect('form-compare', '/api/compare', answer => {
  showAnswer('c', answer);
  const columns = answer.columns || [];
  fillTable($('c-table'), columns, answer.rows || [], columns.length - 1);
});
function usePreset(name) {
  const [span, rows] = PRESETS[name];
  for (let i = 0; i < 5; i++) {
    const [daily, doses, volume] = rows[i] || ['', 4, ''];
    compareForm.elements['daily' + i].value = daily;
    compareForm.elements['doses' + i].value = doses;
    compareForm.elements['volume' + i].value = volume;
  }
  $('c-span').value = span;
  runCompare();
}
document.querySelectorAll('[data-preset]').forEach(button =>
  button.addEventListener('click', () => usePreset(button.dataset.preset)));

const runFind = connect('form-find', '/api/find', answer => {
  showAnswer('f', answer);
  fillTable($('f-map'), answer.map_columns || [], answer.map_rows || [], 1);
  $('f-refinement-note').textContent = answer.refinement_note || '';
  fillTable($('f-refinement'), answer.refinement_columns || [], answer.refinement_rows || [], 2);
  fillTable($('f-windows'), answer.window_columns || [], answer.window_rows || []);
});

const runMeasured = connect('form-measured', '/api/measured', answer => {
  showAnswer('m', answer);
  fillTable($('m-overall'), [], answer.overall || []);
  fillTable($('m-samples'), answer.sample_columns || [], answer.sample_rows || []);
  fillTable($('m-people'), answer.people_columns || [], answer.people_rows || []);
});
$('m-subject').addEventListener('change', runMeasured);

document.querySelectorAll('.legend').forEach(note => {
  note.innerHTML = LEGEND.map(([label, text]) => `<span class="chip ${STATUS[label]}">${label}</span> ${text}`).join(' &nbsp; ');
});

const tabs = [...document.querySelectorAll('[role=tab]')];
const firstOpen = {patient: runPatient, schedule: runSchedule, compare: () => usePreset('split'),
                   find: runFind, measured: runMeasured};
function openTab(name) {
  tabs.forEach(tab => {
    const open = tab.dataset.tab === name;
    tab.setAttribute('aria-selected', open);
    tab.tabIndex = open ? 0 : -1;
  });
  document.querySelectorAll('.tab').forEach(panel => { panel.hidden = panel.id !== 'tab-' + name; });
  history.replaceState(null, '', '#' + name);
  if (firstOpen[name]) {      // each calculation runs once by itself the first time its tab opens
    firstOpen[name]();
    delete firstOpen[name];
  }
}
tabs.forEach(tab => tab.addEventListener('click', () => openTab(tab.dataset.tab)));
$('tabs').addEventListener('keydown', event => {
  const move = {ArrowRight: 1, ArrowLeft: -1}[event.key];
  if (!move) return;
  const next = tabs[(tabs.indexOf(document.activeElement) + move + tabs.length) % tabs.length];
  next.focus();
  openTab(next.dataset.tab);
});
openTab(tabs.some(tab => '#' + tab.dataset.tab === location.hash) ? location.hash.slice(1) : 'patient');

$('stop').addEventListener('click', async () => {
  await fetch('/api/stop', {method: 'POST'}).catch(() => {});
  document.querySelector('main').innerHTML = '<h1>The calculator has stopped</h1>' +
    '<p class="lead">You can close this tab. Run the Python file again to use the calculator.</p>';
});
</script>
</body>
</html>
"""


def web_options(values):
    return ''.join(f'<option>{html.escape(str(value))}</option>' for value in values)


def web_page():
    """The page, with this run's options, findings, verification checks and study graphs filled in."""
    findings, check_rows = '', ''
    for text in FINDINGS:
        label, rest = text.split('  ', 1) if '  ' in text else ('', text)
        findings += f'<li><strong>{html.escape(label)}</strong> {html.escape(rest)}</li>'
    for row in checks:
        check_rows += '<tr>' + ''.join(f'<td>{html.escape(str(row[key]))}</td>'
                                       for key in ('Check', 'Result', 'Status')) + '</tr>'
    graphs = ''.join(f'<figure><img src="/results/{name}" alt="{name}" loading="lazy">'
                     f'<figcaption>{name}</figcaption></figure>' for name in GRAPHS)
    fill = (('__DRUGS__', web_options(DRUGS)), ('__DOSES__', web_options(FREQUENCIES)),
            ('__SUBJECTS__', web_options(range(1, 13))), ('__FINDINGS__', findings),
            ('__CHECKS__', check_rows), ('__GRAPHS__', graphs),
            ('__REFERENCE__', json.dumps([reference['daily_mg'], reference['doses_per_day']])))
    page = WEB_PAGE
    for mark, text in fill:
        page = page.replace(mark, text)
    return page


def picture(fig):
    """A figure as text the page can show (a PNG in base64)."""
    fig.tight_layout()
    image = io.BytesIO()
    fig.savefig(image, format='png', dpi=150, bbox_inches='tight')
    return base64.b64encode(image.getvalue()).decode()


def web_answer(drug_name, age_text, weight_text):
    """What the page shows for one patient: the dosage, its details and the graph."""
    if drug_name not in DRUGS:
        return dict(found=False, headline='Check the numbers',
                    details=['Select a drug from the calculator list.'], graph=None)
    try:
        age, weight = float(age_text), float(weight_text)
    except ValueError:
        age = weight = math.nan
    if not (0 <= age <= 110 and 2 <= weight <= 250):
        return dict(found=False, headline='Check the numbers',
                    details=['Age must be 0 to 110 years and weight 2 to 250 kg.'], graph=None)
    with WEB_LOCK:                      # one calculation at a time: matplotlib is not thread-safe
        result = recommend(drug_name, age, weight)
        graph = None
        if result['run']:
            fig = Figure(figsize=(7.2, 4.6))
            plot_concentration(fig.add_subplot(), [(result['title'], result['run'])], result['title'])
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, 'fig7_patient_dose.png'), bbox_inches='tight')
            graph = picture(fig)
    print(f"Calculator: {drug_name}, age {age:g}, {weight:g} kg -> {result['headline']}")
    return dict(found=result['run'] is not None, headline=result['headline'],
                details=result['details'], graph=graph)


def web_number(form, key, label, low, high, default=None):
    """One number from the page. A ValueError carries the message the page shows."""
    try:
        value = float(form.get(key, default))
    except (TypeError, ValueError):
        value = math.nan
    if not low <= value <= high:
        raise ValueError(f'{label} must be a number from {low:,g} to {high:,g}.')
    return value


def web_parameters(form, length_h, row=''):
    """A schedule from the page. The drug is the handout's unless the page sends other drug values."""
    doses = web_number(form, 'doses' + row, 'Doses per day', 1, 24)
    if doses not in FREQUENCIES:
        raise ValueError('Doses per day must be ' + ', '.join(map(str, FREQUENCIES)) + '.')
    p = Parameters(daily_mg=web_number(form, 'daily' + row, 'Daily dose', 1, 1000000), doses_per_day=int(doses),
                   volume_l=web_number(form, 'volume' + row, 'Blood volume', 0.05, 200),
                   ka_per_h=web_number(form, 'ka', 'Absorption rate', 0.01, 20, base.ka_per_h),
                   half_life_h=web_number(form, 'half_life', 'Half-life', 0.1, 200, base.half_life_h),
                   bioavailability=web_number(form, 'bioavailability', 'Bioavailability', 0.01, 1, base.bioavailability),
                   medicinal_mg_l=web_number(form, 'low', 'Medicinal level', 0.001, 100000, LOW),
                   toxic_mg_l=web_number(form, 'high', 'Toxic level', 0.001, 1000000, HIGH),
                   duration_h=length_h)
    if p.medicinal_mg_l >= p.toxic_mg_l:
        raise ValueError('The medicinal level must be below the toxic level.')
    return p


def level(value):
    """A concentration in the study's style: one decimal, or more digits when it is small."""
    return f'{value:,.1f}' if abs(value) >= 100 else f'{sig(value):g}'


def entry_text(row):
    return 'never' if row['first_entry_h'] is None else f"{row['first_entry_h']:.2f} h"


def verdict(row):
    """A run's status (the Table 9 labels) in plain words."""
    if status(row) == 'n/c' and row['peak_full_mg_l'] < row['medicinal_mg_l']:
        return 'It never reaches the medicinal level in this run, and the level is still changing. Try a longer run.'
    return {'OK+': 'In range from the first time it reaches the medicinal level until the end.',
            'OK': 'The repeating cycle stays in range, but the level dips below the medicinal level during build-up.',
            'low': 'Too low: even the peaks stay below the medicinal level.',
            'dip': 'Too low at the bottom of each cycle: the troughs fall below the medicinal level.',
            'high': 'Too high: the peaks reach the toxic level.',
            'swing': 'Swings too much: the peaks reach the toxic level and the troughs fall below the medicinal level.',
            'n/c': 'Not settled yet: the level has not reached a repeating pattern in this run. Try a longer run.'
            }[status(row)]


def web_schedule(form):
    """Try a schedule (RQ1): one run of the model and its summary."""
    length_h = web_number(form, 'hours', 'Run length', 48, 336, 48)
    if length_h not in (48, 72, 168, 336):
        raise ValueError('Run length must be 48, 72, 168 or 336 hours.')
    p = web_parameters(form, length_h)
    run = simulate(p)
    row, c = summarize(run), run['concentration']
    top = max(range(len(c)), key=c.__getitem__)
    theory = p.bioavailability * p.daily_mg / (24 * p.ke_per_h * p.volume_l)
    facts = [['Each dose', f"{sig(row['mg_per_dose']):,g} mg every {row['interval_h']:g} hours"],
             ['Highest level in the run', f"{level(c[top])} mg/L at {run['time'][top]:.2f} h"],
             ['First reaches the target range', entry_text(row)],
             [f'Hours below / in range / toxic (of {length_h:g})',
              f"{row['below_h']:.1f} / {row['within_h']:.1f} / {row['at_or_above_toxic_h']:.1f}"],
             ['Last cycle: trough to peak',
              f"{level(row['last_cycle_min_mg_l'])} to {level(row['last_cycle_max_mg_l'])} mg/L"],
             ['Last cycle: peak/trough', f"{row['last_cycle_max_mg_l'] / row['last_cycle_min_mg_l']:.3f} "
              f'(the limits allow under {p.toxic_mg_l / p.medicinal_mg_l:.3g})'],
             ['Last cycle: average', f"{level(row['last_cycle_mean_mg_l'])} mg/L"],
             ['Long-run average from the formula', f'{level(theory)} mg/L'],
             ['Last cycle repeats the one before within 0.1%?',
              f"{yes_no(row['last_cycle_converged'])} (change {row['last_cycle_relative_change']:.1%})"],
             ['Drug balance error', f"{row['max_mass_balance_error_mg']:.1e} mg"]]
    every_h = 2.0 if length_h <= 72 else 6.0
    value_rows = []
    for i in range(0, len(c), integer_steps(every_h, p.dt_h)):
        value_rows.append([f"{run['time'][i]:g}", f"{run['intake'][i] if i < len(run['intake']) else 0:,g}",
                           f"{run['gut'][i]:,.1f}", f"{run['plasma'][i]:,.1f}", f"{run['eliminated'][i]:,.1f}",
                           level(c[i]), zone(c[i], p.medicinal_mg_l, p.toxic_mg_l)])
    fig = Figure(figsize=(7.2, 7.2))
    upper, lower = fig.subplots(2, 1)
    plot_concentration(upper, [(schedule_text(row), run)], f'{schedule_text(row)}, {p.volume_l:g} L')
    plot_amounts(lower, run, 'Drug amounts in each part of the model')
    print(f'Schedule: {schedule_text(row)}, {p.volume_l:g} L, {length_h:g} h -> {status(row)}')
    return dict(found=True, status=status(row), headline=schedule_text(row), details=[verdict(row)], facts=facts,
                values_title=f'Values every {every_h:g} hours',
                value_columns=['Time (h)', 'Dose taken (mg)', 'Intestines G (mg)', 'Plasma P (mg)',
                               'Eliminated E (mg)', 'Concentration C (mg/L)', 'Zone'],
                value_rows=value_rows, graph=picture(fig))


def web_compare(form):
    """Compare schedules (RQ2, RQ4): up to five one-week runs, as in Tables 7, 8 and 13."""
    span_h = web_number(form, 'span', 'Graph length', 48, 168, 72)
    runs = [simulate(web_parameters(form, EXTENDED_H, str(i))) for i in range(5) if form.get(f'daily{i}')]
    if not runs:
        raise ValueError('Enter at least one schedule.')
    texts = [(f'{r["p"].daily_mg:,g} mg/day',
              f'{r["p"].doses_per_day} dose{"s" if r["p"].doses_per_day > 1 else ""}/day',
              f'{r["p"].volume_l:.2f} L') for r in runs]
    differs = [i for i in range(3) if len({text[i] for text in texts}) > 1] or [0, 1, 2]
    labels = [', '.join(text[i] for i in differs) for text in texts]
    shared = ', '.join(texts[0][i] for i in range(3) if i not in differs)
    rows = []
    for label, run in zip(labels, runs):
        r = summarize(run)
        rows.append([label, f"{sig(r['mg_per_dose']):,g}", f"{r['interval_h']:g}", level(r['last_cycle_min_mg_l']),
                     level(r['last_cycle_max_mg_l']), f"{r['last_cycle_max_mg_l'] / r['last_cycle_min_mg_l']:.3f}",
                     level(r['last_cycle_mean_mg_l']), f"{r['within_h']:.1f}", entry_text(r), status(r)])
    fig = Figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot()
    plot_concentration(ax, list(zip(labels, runs)), f'First {span_h:g} hours')
    ax.set_xlim(0, span_h)
    print(f'Compare: {len(runs)} schedule{"s" if len(runs) > 1 else ""}')
    return dict(found=True, headline=shared or f'{len(runs)} schedules', graph=picture(fig), rows=rows,
                columns=['Schedule', 'mg per dose', 'Every (h)', 'Trough (mg/L)', 'Peak (mg/L)', 'Peak/trough',
                         'Average (mg/L)', 'Hours in range (of 168)', 'First entry', 'Status'])


def web_find(form):
    """Find schedules (RQ3, RQ4): every tested schedule at one blood volume, as in Tables 9, 12 and 14."""
    volume = web_number(form, 'volume', 'Blood volume', 0.05, 200, 4.60)
    first = web_number(form, 'first', 'Lowest daily total', 1, 1000000, DAILY_TOTALS[0])
    last = web_number(form, 'last', 'Highest daily total', first, 1000000, DAILY_TOTALS[-1])
    step = web_number(form, 'step', 'Step', 1, 1000000, TOTAL_STEP)
    count = math.floor((last - first) / step + 1e-9) + 1
    if count > 61:
        raise ValueError('Test at most 61 daily totals: use a larger step or a smaller range.')
    totals = [first + k * step for k in range(count)]
    tested = {(d, f): summarize(simulate(Parameters(daily_mg=d, doses_per_day=f, volume_l=volume,
                                                    duration_h=EXTENDED_H)))
              for d in totals for f in FREQUENCIES}
    # The Part 5 rule: passing schedules are re-run at REFINED_DT, then the fewest doses and lowest total win.
    checked = {(d, f): summarize(simulate(Parameters(daily_mg=d, doses_per_day=f, volume_l=volume,
                                                     duration_h=EXTENDED_H, dt_h=REFINED_DT)))
               for (d, f), r in tested.items()
               if r['last_cycle_in_range_and_converged'] or r['stays_in_range_after_first_entry']}
    refinement_rows = [[f'{d:,g}', f, status(tested[d, f]), status(r)] for (d, f), r in checked.items()
                       if status(tested[d, f]) != status(r)]
    refinement_note = (f'{len(checked)} candidates were rechecked at {REFINED_DT:g} h. '
                       'The selected schedules use the finer results.')
    if not refinement_rows:
        refinement_note += ' No status labels changed.'
    picks = {}
    for label, key in criteria:
        passing = [r for r in checked.values() if r[key]]
        picks[label] = min(passing, key=lambda r: (r['doses_per_day'], r['daily_mg'])) if passing else None
    details, series = [], []
    for label, r in picks.items():
        if r is None:
            details.append(f'{label}: no tested schedule.')
            continue
        details.append(f"{label}: {schedule_text(r)} ({sig(r['mg_per_dose']):,g} mg every {r['interval_h']:g} hours, "
                       f"repeats between {level(r['last_cycle_min_mg_l'])} and {level(r['last_cycle_max_mg_l'])} mg/L).")
        series.append((f'{label}: {schedule_text(r)}',
                       simulate(Parameters(daily_mg=r['daily_mg'], doses_per_day=r['doses_per_day'], volume_l=volume,
                                           duration_h=EXTENDED_H, dt_h=REFINED_DT))))
    window_rows = []
    for f in FREQUENCIES:
        r = tested[first, f]
        window_rows.append([f, f"{r['last_cycle_max_mg_l'] / r['last_cycle_min_mg_l']:.3f}",
                            estimated_dose_window(r)])
    graph = None
    if series:
        fig = Figure(figsize=(7.2, 4.6))
        plot_concentration(fig.add_subplot(), series, f'Selected tested schedules at {volume:g} L')
        graph = picture(fig)
    best = picks[criteria[0][0]]
    headline = schedule_text(best) if best else 'No tested schedule fits'
    print(f'Find: {volume:g} L -> {headline}')
    return dict(found=best is not None, headline=headline, details=details, graph=graph,
                map_columns=['Daily total (mg)'] + [f'{f}/day' for f in FREQUENCIES],
                map_rows=[[f'{d:,g}'] + [status(tested[d, f]) for f in FREQUENCIES] for d in totals],
                refinement_note=refinement_note,
                refinement_columns=['Daily total (mg)', 'Doses/day', '0.01 h', '0.005 h'] if refinement_rows else [],
                refinement_rows=refinement_rows,
                window_columns=['Doses/day', 'Peak/trough', f'Daily total at {volume:g} L (mg/day)'],
                window_rows=window_rows)


def web_measured(form):
    """Measured data (RQ6): one person's Theoph measurements against the PART 9b prediction."""
    subject = web_number(form, 'subject', 'Person', 1, 12, 1)
    if not subject.is_integer():
        raise ValueError('Person must be a whole number from 1 to 12.')
    subject = int(subject)
    samples = pd.read_csv(os.path.join(OUT, 'theoph_observed_vs_predicted.csv'))
    people = pd.read_csv(os.path.join(OUT, 'theoph_subject_metrics.csv'))
    group = samples[samples['Subject'] == subject].sort_values('Time_h')
    person = people[people['Subject'] == subject].iloc[0]
    weight, dose, start = (float(group.iloc[0][key]) for key in
                           ('Weight_kg', 'Single_dose_mg', 'Initial_concentration_mg_L'))
    fixed = {row['Parameter']: row['Value'] for row in theoph_comparison['parameters']}
    p = Parameters(daily_mg=dose, doses_per_day=1, volume_l=fixed['Distribution volume per kg'] * weight,
                   ka_per_h=fixed['Absorption rate'], half_life_h=fixed['Half-life'], duration_h=25.,
                   single_dose=True, dt_h=fixed['Numerical time step'], medicinal_mg_l=10., toxic_mg_l=20.)
    run = simulate(p)
    curve = [c + start * math.exp(-p.ke_per_h * t) for t, c in zip(run['time'], run['concentration'])]
    fig = Figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot()
    ax.plot(run['time'], curve, color=COLORS[0], label='Model with fixed published values')
    ax.scatter(group['Time_h'], group['Observed_mg_L'], s=30, color=COLORS[1], zorder=3, label='Measured serum')
    ax.set(title=f'Person {subject}: {weight:g} kg, one dose of {dose:,.0f} mg', xlabel='Time after one dose (h)',
           ylabel='Concentration (mg/L)', xlim=(0, 25), ylim=(0, None))
    ax.legend(loc='best')
    overall = theoph_comparison['metrics'][0]
    print(f'Measured data: person {subject}')
    return dict(
        found=True, headline=f"Average error {person['Post-dose MAE (mg/L)']:.4f} mg/L", graph=picture(fig),
        details=[f'Weight {weight:g} kg, one dose of {dose:,.1f} mg.',
                 f"Root mean squared error {person['Post-dose RMSE (mg/L)']:.4f} mg/L over 10 samples after the dose.",
                 f"Highest measured {person['Observed sampled peak (mg/L)']:.2f} mg/L; "
                 f"highest predicted {person['Predicted sampled peak (mg/L)']:.2f} mg/L."],
        overall=[['Average error (MAE)', f"{overall['MAE (mg/L)']:.4f} mg/L"],
                 ['Root mean squared error (RMSE)', f"{overall['RMSE (mg/L)']:.4f} mg/L"],
                 ['Average bias (predicted minus measured)', f"{overall['Mean bias (mg/L)']:+.4f} mg/L"],
                 ['Descriptive R squared', f"{overall['R squared (descriptive)']:.4f}"],
                 ['What this does not show', 'One dose only: it does not test repeating doses, children or '
                                             'individual prescriptions.']],
        sample_columns=['Time (h)', 'Measured (mg/L)', 'Predicted (mg/L)', 'Predicted minus measured (mg/L)'],
        sample_rows=[[f"{row['Time_h']:g}", f"{row['Observed_mg_L']:.2f}", f"{row['Predicted_mg_L']:.2f}",
                      f"{row['Residual_predicted_minus_observed_mg_L']:+.2f}"] for _, row in group.iterrows()],
        people_columns=['Person', 'Weight (kg)', 'Dose (mg)', 'Average error (mg/L)', 'RMSE (mg/L)'],
        people_rows=[[int(row['Subject']), f"{row['Weight (kg)']:g}", f"{row['Single dose (mg)']:,.1f}",
                      f"{row['Post-dose MAE (mg/L)']:.4f}", f"{row['Post-dose RMSE (mg/L)']:.4f}"]
                     for _, row in people.iterrows()])


WEB_CALCULATIONS = {'/api/schedule': web_schedule, '/api/compare': web_compare,
                    '/api/find': web_find, '/api/measured': web_measured}


class CalculatorRequests(BaseHTTPRequestHandler):
    """The browser's requests: the page, a calculation, a study graph, or stop."""

    def do_GET(self):
        url = urlparse(self.path)
        graph = url.path.removeprefix('/results/')
        form = {key: values[0] for key, values in parse_qs(url.query, keep_blank_values=True).items()}
        if url.path == '/':
            self.reply('text/html; charset=utf-8', web_page().encode())
        elif url.path == '/api/dose':
            answer = web_answer(form.get('drug', ''), form.get('age', ''), form.get('weight', ''))
            self.reply('application/json', json.dumps(answer).encode())
        elif url.path in WEB_CALCULATIONS:
            try:
                with WEB_LOCK:          # one calculation at a time: matplotlib is not thread-safe
                    answer = WEB_CALCULATIONS[url.path](form)
            except ValueError as problem:
                answer = dict(found=False, headline='Check the numbers', details=[str(problem)])
            self.reply('application/json', json.dumps(answer).encode())
        elif graph in GRAPHS:           # only the graphs this run saved
            with open(os.path.join(OUT, graph), 'rb') as file:
                self.reply('image/png', file.read())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != '/api/stop':
            self.send_error(404)
            return
        self.reply('text/plain', b'stopped')
        threading.Thread(target=self.server.shutdown).start()   # ends serve_forever below

    def reply(self, content_type, body):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass                            # the terminal shows one line per calculation instead


def start_web_server():
    """The calculator's web server, for this computer only (127.0.0.1); None if it cannot start."""
    for port in (WEB_PORT, 0):          # 0: any free port when WEB_PORT is busy
        try:
            return ThreadingHTTPServer(('127.0.0.1', port), CalculatorRequests)
        except OSError:
            pass
    return None


WEB_LOCK = threading.Lock()
if isinstance(globals().get('calculator_server'), ThreadingHTTPServer):   # notebook re-run:
    calculator_server.shutdown()                                          # stop the last run's page
    calculator_server.server_close()
calculator_server = start_web_server() if WEB_APP and not NO_UI else None
if NO_UI:
    print('Calculator skipped (--no-ui); the complete study has been saved.')
elif calculator_server is None:
    typed_calculator()
else:
    url = f'http://127.0.0.1:{calculator_server.server_port}'
    print(f'The calculator web application opens in your browser: {url}\n'
          '(if it does not, Cmd+click or Ctrl+click that link).\n'
          "Its tabs run the study's calculations. Each one prints a line here.")
    webbrowser.open(url)
    if 'ipykernel' in sys.modules:      # notebook: the page keeps working after this cell ends
        threading.Thread(target=calculator_server.serve_forever, daemon=True).start()
        print('The page works until "Stop calculator" is pressed or the notebook restarts.')
    else:
        print('To end the program, press "Stop calculator" on the page, or Ctrl+C here.')
        try:
            calculator_server.serve_forever()
        except KeyboardInterrupt:       # Ctrl+C in the terminal
            pass
        calculator_server.server_close()
        print('Calculator stopped.')
print(f'\nAll tables (.csv) and graphs (.png) are saved in: {os.path.abspath(OUT)}')
