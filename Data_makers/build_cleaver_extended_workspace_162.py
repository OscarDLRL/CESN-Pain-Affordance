#!/usr/bin/env python3

"""
Extended Workspace Dataset V2.

Reuses the validated CLEAVER dataset generator.

Robot workspace: 16 x 16 cm.
Object workspace: 12 x 12 cm.

81 paired configurations.
162 SAFE/PAIN trajectories.

The original dataset remains unchanged.
"""

import sys
import argparse
from pathlib import Path

import numpy as np

from Data_makers import (
    build_cleaver_dataset_162_fixed as generator
)


# ==================================================
# EXTENDED WORKSPACE CONFIGURATION
# ==================================================

START_CENTER = np.array([
    -.05,
    -.05,
     .35,
])

START_OFFSETS = [
    (x, y)
    for y in (.08, 0., -.08)
    for x in (-.08, 0., .08)
]

OBJECT_OFFSETS = [
    (x, y)
    for y in (.06, 0., -.06)
    for x in (-.09, -.03, .03)
]

OUT = Path(
    "datasets/cleaver_extended_workspace_162"
)


# ==================================================
# APPLY NEW CONFIGURATION
# ==================================================

generator.START_CENTER = START_CENTER.copy()

generator.START_OFFSETS = START_OFFSETS

generator.OBJECT_OFFSETS = OBJECT_OFFSETS


# ==================================================
# DISPLAY COORDINATES
# ==================================================

def print_configuration():

    print("\n" + "=" * 65)
    print("CLEAVER EXTENDED WORKSPACE V2")
    print("=" * 65)

    print("\nRobot initial positions (meters):")

    for i, (x, y) in enumerate(START_OFFSETS):

        position = START_CENTER + np.array([
            x, y, 0.
        ])

        print(
            f"R{i + 1}: "
            f"{np.round(position, 4)}"
        )

    print("\nObject positions (meters):")

    for i, (x, y) in enumerate(OBJECT_OFFSETS):

        position = generator.v4.ORIGIN + np.array([
            x, y, 0.
        ])

        print(
            f"K{i + 1}: "
            f"{np.round(position, 4)}"
        )

    print("\nOutput directory:", OUT)
    print("=" * 65 + "\n")


# ==================================================
# MAIN
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Test object and robot workspace extremes."
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        type=int
    )

    parser.add_argument(
        "--show-config",
        action="store_true"
    )

    args = parser.parse_args()

    print_configuration()

    if args.show_config:
        return

    command = [
        sys.argv[0],
        "--out",
        str(OUT),
    ]

    if args.resume:
        command.append("--resume")

    if args.pairs is not None:

        command.extend([
            "--pairs",
            *map(str, args.pairs)
        ])

    elif args.pilot:

        # Central robot with all nine object positions.
        object_test = [
            obj * 9 + 4
            for obj in range(9)
        ]

        # Central object with all nine robot positions.
        robot_test = [
            4 * 9 + robot
            for robot in range(9)
        ]

        # Extreme corner combinations.
        corner_test = [
            obj * 9 + robot
            for obj in (0, 2, 6, 8)
            for robot in (0, 2, 6, 8)
        ]

        pilot_pairs = sorted(set(
            object_test
            + robot_test
            + corner_test
        ))

        print(
            "Pilot pairs:",
            pilot_pairs
        )

        print(
            "Number of pilot pairs:",
            len(pilot_pairs)
        )

        command.extend([
            "--pairs",
            *map(str, pilot_pairs)
        ])

    sys.argv = command

    generator.main()


if __name__ == "__main__":
    main()

