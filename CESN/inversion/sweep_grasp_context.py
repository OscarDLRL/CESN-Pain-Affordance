from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = (
    ROOT
    / "CESN"
    / "outputs"
    / "rgb_grasp_context_cesn_v1"
    / "context_inversion"
)

FRACTIONS = [
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
]


def main():
    rows = []

    for pct in FRACTIONS:
        print()
        print(
            "=" * 100
        )
        print(
            f"RUNNING HISTORY = "
            f"{pct}%"
        )
        print(
            "=" * 100
        )

        cmd = [
            sys.executable,
            "-m",
            "CESN.inversion.invert_grasp_context",
            "--history",
            str(pct),
        ]

        subprocess.run(
            cmd,
            check=True,
        )

        path = (
            OUT_DIR
            / f"inversion_{pct}pct.json"
        )

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        rows.append(
            {
                "history_percent":
                    pct,

                "task_accuracy":
                    data[
                        "task_accuracy"
                    ],

                "mean_grasp_error_mm":
                    data[
                        "mean_grasp_error_mm"
                    ],

                "median_grasp_error_mm":
                    data[
                        "median_grasp_error_mm"
                    ],

                "max_grasp_error_mm":
                    data[
                        "max_grasp_error_mm"
                    ],
            }
        )

    summary_path = (
        OUT_DIR
        / "history_sweep_summary.json"
    )

    csv_path = (
        OUT_DIR
        / "history_sweep_summary.csv"
    )

    summary_path.write_text(
        json.dumps(
            rows,
            indent=2,
        ),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        "=" * 100
    )
    print(
        "HISTORY SWEEP SUMMARY"
    )
    print(
        "=" * 100
    )

    for row in rows:
        print(
            f"{row['history_percent']:>3d}%"
            f" | task="
            f"{row['task_accuracy'] * 100:>6.2f}%"
            f" | meanXY="
            f"{row['mean_grasp_error_mm']:>8.2f} mm"
            f" | medianXY="
            f"{row['median_grasp_error_mm']:>8.2f} mm"
        )

    print()
    print(
        summary_path.resolve()
    )
    print(
        csv_path.resolve()
    )
    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
