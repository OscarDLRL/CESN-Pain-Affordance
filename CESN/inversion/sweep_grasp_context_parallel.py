from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
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


def run_one_history(pct: int):
    """
    Run one complete inversion sweep for one history percentage.

    Each worker is a separate Python process. CUDA_VISIBLE_DEVICES
    is inherited, so all workers share the same visible RTX 3090.
    """

    env = os.environ.copy()

    # Prevent each subprocess from also spawning many CPU threads.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    # Helps reduce CUDA allocator fragmentation when several
    # independent PyTorch processes share one GPU.
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    cmd = [
        sys.executable,
        "-m",
        "CESN.inversion.invert_grasp_context",
        "--history",
        str(pct),
    ]

    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"History {pct}% failed.\n\n"
            f"{completed.stdout}"
        )

    json_path = (
        OUT_DIR
        / f"inversion_{pct}pct.json"
    )

    if not json_path.exists():
        raise RuntimeError(
            f"History {pct}% completed but "
            f"{json_path} was not created.\n\n"
            f"{completed.stdout}"
        )

    data = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    return {
        "history_percent":
            int(pct),

        "task_accuracy":
            float(
                data[
                    "task_accuracy"
                ]
            ),

        "mean_grasp_error_mm":
            float(
                data[
                    "mean_grasp_error_mm"
                ]
            ),

        "median_grasp_error_mm":
            float(
                data[
                    "median_grasp_error_mm"
                ]
            ),

        "max_grasp_error_mm":
            float(
                data[
                    "max_grasp_error_mm"
                ]
            ),

        "log":
            completed.stdout,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Number of history percentages evaluated concurrently "
            "on the same visible GPU. Start with 4 on RTX 3090."
        ),
    )

    parser.add_argument(
        "--histories",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional custom history percentages. "
            "Example: --histories 5 10 15 20 25 30"
        ),
    )

    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers must be >= 1"
        )

    histories = (
        FRACTIONS
        if not args.histories
        else sorted(
            set(
                int(x)
                for x in args.histories
            )
        )
    )

    for pct in histories:
        if pct <= 0 or pct > 100:
            raise ValueError(
                f"Invalid history percentage: {pct}"
            )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 104)
    print(
        "PARALLEL CESN GRASP-CONTEXT HISTORY SWEEP"
    )
    print("=" * 104)
    print(
        "Histories:",
        histories,
    )
    print(
        "Parallel workers:",
        args.workers,
    )
    print(
        "Each worker runs a complete 22-trajectory "
        "context-inversion experiment."
    )
    print(
        "All workers share the GPU selected by CUDA_VISIBLE_DEVICES."
    )
    print("=" * 104)
    print()

    results = []

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:

        future_to_pct = {
            executor.submit(
                run_one_history,
                pct,
            ):
                pct
            for pct
            in histories
        }

        for future in as_completed(
            future_to_pct
        ):
            pct = future_to_pct[
                future
            ]

            try:
                row = future.result()

            except Exception as exc:
                print()
                print(
                    f"[FAIL] history={pct}%"
                )
                print(
                    str(exc)
                )

                # Cancel pending work before exiting.
                for other in future_to_pct:
                    other.cancel()

                raise

            results.append(
                row
            )

            print(
                f"[DONE] {pct:>3d}%"
                f" | TASK="
                f"{row['task_accuracy'] * 100:>6.2f}%"
                f" | meanXY="
                f"{row['mean_grasp_error_mm']:>8.2f} mm"
                f" | medianXY="
                f"{row['median_grasp_error_mm']:>8.2f} mm"
                f" | maxXY="
                f"{row['max_grasp_error_mm']:>8.2f} mm"
            )

    results.sort(
        key=lambda row:
            row[
                "history_percent"
            ]
    )

    # Save compact summary without the captured console logs.
    compact_rows = [
        {
            key: value
            for key, value
            in row.items()
            if key != "log"
        }
        for row
        in results
    ]

    summary_json = (
        OUT_DIR
        / "parallel_history_sweep_summary.json"
    )

    summary_csv = (
        OUT_DIR
        / "parallel_history_sweep_summary.csv"
    )

    summary_json.write_text(
        json.dumps(
            compact_rows,
            indent=2,
        ),
        encoding="utf-8",
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "history_percent",
                "task_accuracy",
                "mean_grasp_error_mm",
                "median_grasp_error_mm",
                "max_grasp_error_mm",
            ],
        )

        writer.writeheader()
        writer.writerows(
            compact_rows
        )

    print()
    print("=" * 104)
    print(
        "PARALLEL HISTORY SWEEP SUMMARY"
    )
    print("=" * 104)

    for row in compact_rows:
        print(
            f"{row['history_percent']:>3d}%"
            f" | TASK="
            f"{row['task_accuracy'] * 100:>6.2f}%"
            f" | meanXY="
            f"{row['mean_grasp_error_mm']:>8.2f} mm"
            f" | medianXY="
            f"{row['median_grasp_error_mm']:>8.2f} mm"
            f" | maxXY="
            f"{row['max_grasp_error_mm']:>8.2f} mm"
        )

    print()
    print(
        "JSON:",
        summary_json.resolve(),
    )
    print(
        "CSV :",
        summary_csv.resolve(),
    )
    print("=" * 104)


if __name__ == "__main__":
    main()
