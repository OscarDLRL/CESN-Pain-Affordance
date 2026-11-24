from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

SOURCE = (
    ROOT
    / "CESN/outputs/reports/paired_method_selection/detail.csv"
)

REPORT = ROOT / "CESN/outputs/final_lbfgs/final_report"

FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Selected method: discrete hypotheses + L-BFGS
# Noise-free observations only
# -------------------------------------------------

df = pd.read_csv(SOURCE)

df = df[
    (df["method_label"] == "A_discrete_lbfgs60")
    & np.isclose(df["noise_sigma"], 0.0)
].copy()

if df.empty:
    raise RuntimeError("No matching experimental results found.")

if df.duplicated(["trajectory", "fraction"]).any():
    raise RuntimeError("Duplicated trajectory/fraction conditions.")

THRESHOLDS = [5, 10, 20]
SUCCESS_RATE = 90.0

# -------------------------------------------------
# Minimum sustained observation fraction
# -------------------------------------------------

def minimum_fraction(data, tolerance):
    grouped = (
        data.groupby("fraction")["xy_error_mm"]
        .apply(lambda x: 100.0 * (x <= tolerance).mean())
        .sort_index()
    )

    fractions = grouped.index.to_numpy()
    rates = grouped.to_numpy()

    # First fraction that satisfies the criterion
    # at every subsequent evaluated fraction.
    for i, fraction in enumerate(fractions):
        if np.all(rates[i:] >= SUCCESS_RATE):
            return float(fraction)

    return np.nan


groups = [
    ("Overall", df),
    ("Pick", df[df["true_task"] == "pick"]),
    ("Push", df[df["true_task"] == "push"]),
    ("Interpolation", df[df["region"] == "interpolation"]),
    ("Extrapolation", df[df["region"] == "extrapolation"]),
]

rows = []

for name, data in groups:
    row = {
        "Evaluation": name,
        "Trajectories": data["trajectory"].nunique(),
    }

    for tolerance in THRESHOLDS:
        row[f"XY error <= {tolerance} mm"] = minimum_fraction(
            data, tolerance
        )

    rows.append(row)

result = pd.DataFrame(rows)

csv_path = TABLES / "16_minimum_context_required.csv"
result.to_csv(csv_path, index=False)

# -------------------------------------------------
# Scientific table figure
# -------------------------------------------------

def format_fraction(value):
    if pd.isna(value):
        return "Not sustained"
    return f"{value * 100:.0f}%"

headers = [
    "Evaluation",
    "N",
    "Error <= 5 mm",
    "Error <= 10 mm",
    "Error <= 20 mm",
]

cell_data = []

for _, row in result.iterrows():
    cell_data.append([
        row["Evaluation"],
        str(int(row["Trajectories"])),
        format_fraction(row["XY error <= 5 mm"]),
        format_fraction(row["XY error <= 10 mm"]),
        format_fraction(row["XY error <= 20 mm"]),
    ])

fig, ax = plt.subplots(figsize=(11.5, 4.1))
ax.axis("off")

fig.text(
    0.5, 0.94,
    "Minimum Observation Required for Spatial Context Recovery",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
)

fig.text(
    0.5, 0.865,
    "Method A + L-BFGS | No observation noise | Cube size: 40 mm",
    ha="center",
    fontsize=10,
)

table = ax.table(
    cellText=cell_data,
    colLabels=headers,
    cellLoc="center",
    colLoc="center",
    colWidths=[0.26, 0.08, 0.22, 0.22, 0.22],
    bbox=[0.035, 0.22, 0.93, 0.56],
)

table.auto_set_font_size(False)
table.set_fontsize(10.5)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#D5DCE5")
    cell.set_linewidth(0.6)

    if r == 0:
        cell.set_facecolor("#24364B")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")

    elif r % 2 == 0:
        cell.set_facecolor("#F1F5F9")

    if r > 0 and c == 3:
        cell.get_text().set_weight("bold")

fig.text(
    0.5, 0.145,
    "Minimum evaluated fraction meeting >=90% localization success "
    "at that and all later evaluated fractions.",
    ha="center",
    fontsize=9,
)

fig.text(
    0.5, 0.09,
    "Not sustained = criterion not maintained through 100% observation. "
    "XY error measures cube-center localization, not gripper error.",
    ha="center",
    fontsize=8.5,
)

png_path = FIGURES / "16_minimum_context_required.png"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("\nMINIMUM TRAJECTORY OBSERVATION")
print(result.to_string(index=False))
print("\nSaved:")
print(png_path)
print(csv_path)
