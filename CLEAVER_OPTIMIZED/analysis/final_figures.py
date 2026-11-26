"""All final figures -> outputs/final/{dataset,experiments}/.

dataset/01_knife_positions.png       one simulator image with the 9 cleaver positions (split-coloured)
dataset/02_robot_initial_positions.png  the 9 robot start positions (fixed cleaver)
experiments/01_reconstruction.png    real vs autoencoder reconstruction (test pair, SAFE and PAIN)
experiments/02_context_history.png   SAFE/PAIN accuracy as the history grows
experiments/03_computational_cost.png  cost per decision of every method
experiments/04_methods_comparison.png  classification-only comparison

Reads outputs/final/results/*.csv (experiments.final_evaluation) and the checkpoints.

    python -m CLEAVER_OPTIMIZED.analysis.final_figures
"""
import csv
import json

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from CLEAVER_OPTIMIZED import config as cfg  # noqa: E402
from CLEAVER_OPTIMIZED import data  # noqa: E402
from CLEAVER_OPTIMIZED.autoencoder.model import load_checkpoint  # noqa: E402
from CLEAVER_OPTIMIZED.perception import GoalPerception, locate  # noqa: E402

FINAL = cfg.OUT / "final"
RES = FINAL / "results"
# reference categorical palette (light mode), fixed order; text never wears series colour
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600"]
TEXT, TEXT2, GRID, NEUTRAL = "#191919", "#666666", "#e6e6e6", "#9c9c9c"
FIXED_START = 4      # R5 (centre) for the knife-position figure
FIXED_OBJECT = 4     # K5 (centre) for the robot-start figure
RECON_PAIR = 40      # test pair: object 4 (K5), start 4 (R5)
RECON_FRAMES = [10, 60, 100, 130, 180, 222]


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": NEUTRAL, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "text.color": TEXT, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 4, "figure.dpi": 100, "savefig.dpi": 300,
        "savefig.bbox": "tight", "legend.frameon": False})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def split_of(pair):
    return "train" if pair in cfg.TRAIN else "validation" if pair in cfg.VALIDATION else "test"


# ------------------------------------------------------------------ dataset figures
def knife_positions(out):
    """3x3 grid: one reference image per cleaver position (fixed robot start)."""
    with h5py.File(cfg.RGB, "r") as f:
        rows = {int(f[k].attrs["object_id"]): (f[k]["rgb"][cfg.REFERENCE_FRAME], f[k]["knife_xy"][:], int(f[k].attrs["pair_id"]))
                for k in f if k.startswith("trajectory_") and int(f[k].attrs["start_id"]) == FIXED_START
                and int(f[k].attrs["label"]) == 0}
    grid_figure(rows, "K", f"Cleaver positions K1–K9 (robot start R{FIXED_START + 1}); training role in brackets",
                out / "01_knife_positions.png", lambda r: split_of(r[2]))
    with open(out / "knife_positions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "object_id", "split", "knife_x_m", "knife_y_m"])
        for o, (_, xy, pair) in sorted(rows.items()):
            w.writerow([f"K{o + 1}", o, split_of(pair), *xy[:2]])


def grid_figure(rows, prefix, title, path, role=None, ncols=3):
    """rows: {index: (rgb, xy_m, ...)} -> grid of reference images titled with coordinates."""
    n = len(rows)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.13 * ncols, 2.2 * nrows))
    for ax, (i, r) in zip(np.atleast_1d(axes).flat, sorted(rows.items())):
        ax.imshow(r[0])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        xy = np.asarray(r[1])[:2] * 100
        extra = f"  [{role(r)}]" if role else ""
        ax.set_title(f"{prefix}{i + 1}  ({xy[0]:.1f}, {xy[1]:.1f}) cm{extra}", fontsize=8)
    for ax in list(np.atleast_1d(axes).flat)[n:]:
        ax.axis("off")
    fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def robot_positions(out):
    with h5py.File(cfg.RGB, "r") as f:
        rows = {int(f[k].attrs["start_id"]): (f[k]["rgb"][cfg.REFERENCE_FRAME], f[k]["requested_start_xyz"][:])
                for k in f if k.startswith("trajectory_") and int(f[k].attrs["object_id"]) == FIXED_OBJECT
                and int(f[k].attrs["label"]) == 0}
    fig, axes = plt.subplots(3, 3, figsize=(6.4, 6.6))
    for s, ax in enumerate(axes.flat):
        img, xyz = rows[s]
        ax.imshow(img)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"R{s + 1}  ({xyz[0] * 100:.0f}, {xyz[1] * 100:.0f}, {xyz[2] * 100:.0f}) cm", fontsize=8)
    fig.suptitle(f"Robot initial positions R1–R9 (cleaver at K{FIXED_OBJECT + 1})", x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "02_robot_initial_positions.png")
    plt.close(fig)


# ------------------------------------------------------------------ experiment figures
@torch.no_grad()
def reconstruction(out, source=None, pair=RECON_PAIR, title=None):
    source = source or cfg.RGB
    ae, _ = load_checkpoint(cfg.OUT / "autoencoder" / cfg.AE_RUN / "autoencoder.pt", "cuda")
    rows = []
    with h5py.File(source, "r") as f:
        for label, name in ((0, "SAFE"), (1, "PAIN")):
            rgb = f[f"trajectory_{2 * pair + label:03d}"]["rgb"]
            ref = data.to_tensor(rgb[cfg.REFERENCE_FRAME:cfg.REFERENCE_FRAME + 1], "cuda").expand(len(RECON_FRAMES), -1, -1, -1)
            y = data.to_tensor(rgb[RECON_FRAMES], "cuda")
            rec = ae(ref, y)["frame_recon"].clamp(0, 1)
            rows.append((f"{name} real", y))
            rows.append((f"{name} reconstruction", rec))
    fig, axes = plt.subplots(len(rows), len(RECON_FRAMES), figsize=(1.55 * len(RECON_FRAMES), 1.6 * len(rows)))
    for r, (name, img) in enumerate(rows):
        a = img.float().cpu().numpy().transpose(0, 2, 3, 1)
        for c, t in enumerate(RECON_FRAMES):
            ax = axes[r, c]
            ax.imshow(a[c])
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"frame {t}", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.suptitle(title or f"Autoencoder reconstruction, test pair {pair} (unseen cleaver position K5)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "01_reconstruction.png")
    plt.close(fig)


def contact_step(source=None, pairs=None):
    source = source or cfg.RGB
    with h5py.File(source, "r") as f:
        steps = []
        for k in f:
            if k.startswith("trajectory_") and (pairs is None or int(f[k].attrs["pair_id"]) in pairs) and int(f[k].attrs["label"]) == 1:
                c = f[k]["contact_impulses"][:]
                steps.append(int(np.flatnonzero(c[:, 0] + c[:, 1] > 1e-4)[0]) - cfg.REFERENCE_FRAME)
    return float(np.median(steps))


def mark_contact(ax, step, top=False):
    ax.axvline(step, color=NEUTRAL, lw=1, ls="--", zorder=0)
    ax.text(step, 0.98 if top else 0.02, " first edge contact\n (median, nominal)", transform=ax.get_xaxis_transform(),
            fontsize=7, color=TEXT2, va="top" if top else "bottom",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))


TEST_SUB = "Test split (36 trajectories, 2 unseen cleaver positions)"
INTERP_SUB = "Interpolation (48 trajectories, 4 off-grid cleaver positions; fitted on all 9 grid positions)"


def context_history(out, contact, res=RES, subtitle=TEST_SUB, name="02_context_history.png"):
    rows=read_csv(res/"history.csv")
    fig, ax=plt.subplots(figsize=(6.4,3.4))
    for i,k in enumerate((1,5,10)):
        rr=[r for r in rows if int(r['obs_every'])==k]
        ax.plot([int(r['frames']) for r in rr],[float(r['accuracy_pct']) for r in rr],'-o',color=SERIES[i],label=f'every {k} step(s)')
    ax.set_xscale('log'); ax.set_xlabel('Elapsed sequence steps (log scale)'); ax.set_ylabel('SAFE/PAIN accuracy (%)'); ax.set_ylim(0,105)
    mark_contact(ax,contact); ax.legend(); ax.set_title(subtitle,loc='left'); fig.tight_layout(); fig.savefig(out/name); plt.close(fig)

NAMES={'cesn_batch':'CESN batch inversion','cesn_incremental':'CESN incremental inversion','nearest':'Nearest trajectory','linear':'Linear ridge'}

def computational_cost(out):
    rows = read_csv(RES / 'cost.csv')
    pick = [
        r for r in rows
        if int(r['frames']) == 111
        and int(r['obs_every']) == 5
    ]

    if len(pick) != 4:
        raise ValueError('Regenerate classification-only cost.csv first')

    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    y = np.arange(len(pick))[::-1]
    ms = [float(r['ms']) for r in pick]

    ax.barh(
        y, ms,
        color=[
            SERIES[0] if r['method'].startswith('cesn')
            else NEUTRAL for r in pick
        ],
        height=0.65
    )

    ax.set_yticks(y, [NAMES[r['method']] for r in pick])

    # Linear scale
    ax.set_xscale('linear')
    ax.set_xlim(0, max(55, max(ms) * 1.2))
    ax.set_xticks(np.arange(0, 51, 10))
    ax.minorticks_off()

    # Control period
    ax.axvline(
        50, color=TEXT2, ls='--', lw=1.2,
        label='20 Hz control period (50 ms)'
    )

    # Value labels
    for yi, v in zip(y, ms):
        ax.text(
            v + 0.7, yi, f'{v:.2f} ms',
            va='center', fontsize=8
        )

    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Median inference latency (ms)')
    ax.set_title('Computational cost per decision', loc='left')
    ax.legend(loc='lower right', fontsize=8)

    fig.tight_layout()
    fig.savefig(out / '03_computational_cost.png')
    plt.close(fig)

    # Closed-loop deployment latency
    p = RES / 'deployment_latency.json'

    if p.exists():
        d = json.loads(p.read_text())
        fig, ax = plt.subplots(figsize=(6, 2.5))

        vals = [
            d['ms_per_step'],
            d['ms_per_observation']
        ]

        ax.barh([1, 0], vals, color=SERIES[0])
        ax.set_yticks(
            [1, 0],
            ['Mean control step',
             'Observation incl. dynamic encoder']
        )

        ax.set_xlabel('Mean detector latency (ms)')
        ax.set_title(
            'Closed-loop deployment (rendering and reset excluded)',
            loc='left'
        )

        fig.tight_layout()
        fig.savefig(out / '05_deployment_latency.png')
        plt.close(fig)
def methods_comparison(out, contact, res=RES, subtitle=TEST_SUB, name="04_methods_comparison.png"):
    rows=read_csv(res/'methods.csv'); fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
    for ax,k in zip(axes,(1,5,10)):
        for i,(m,label) in enumerate(NAMES.items()):
            rr=[r for r in rows if r['method']==m and int(r['obs_every'])==k]
            ax.plot([int(r['frames']) for r in rr],[float(r['accuracy_pct']) for r in rr],marker='o',color=SERIES[i],label=label,ls='--' if m=='cesn_incremental' else '-')
        ax.set_xscale('log'); ax.set_ylim(0,105); ax.set_title(f'Observation every {k} step(s)'); ax.set_xlabel('Elapsed steps'); mark_contact(ax,contact)
    axes[0].set_ylabel('SAFE/PAIN accuracy (%)'); handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=2); fig.suptitle(subtitle); fig.tight_layout(rect=(0,.15,1,.93)); fig.savefig(out/name); plt.close(fig)


def offgrid_positions(out):
    """2x2 grid of the 4 off-grid cleaver positions (fixed robot start)."""
    with h5py.File(cfg.OFFGRID, "r") as f:
        rows = {int(f[k].attrs["object_id"]): (f[k]["rgb"][cfg.REFERENCE_FRAME], f[k]["knife_xy"][:])
                for k in f if k.startswith("trajectory_") and int(f[k].attrs["start_id"]) == FIXED_START
                and int(f[k].attrs["label"]) == 0}
    grid_figure(rows, "I", f"Off-grid cleaver positions I1–I4\n(robot start R{FIXED_START + 1}; never used for fitting)",
                out / "01_offgrid_knife_positions.png", ncols=2)


INTERP_PAIR = 9 * 2 + FIXED_START  # off-grid position I3, robot start R5


def main():
    style()
    ds, ex = FINAL / "dataset", FINAL / "experiments"
    ds.mkdir(parents=True, exist_ok=True)
    ex.mkdir(parents=True, exist_ok=True)
    knife_positions(ds)
    robot_positions(ds)
    contact = contact_step(cfg.RGB, set(cfg.TEST))
    reconstruction(ex)
    context_history(ex, contact)
    computational_cost(ex)
    methods_comparison(ex, contact)
    ires = FINAL / "interpolation" / "results"
    if (ires / "methods.csv").exists():
        ids, iex = FINAL / "interpolation" / "dataset", FINAL / "interpolation" / "experiments"
        ids.mkdir(parents=True, exist_ok=True)
        iex.mkdir(parents=True, exist_ok=True)
        offgrid_positions(ids)
        icontact = contact_step(cfg.OFFGRID)
        reconstruction(iex, cfg.OFFGRID, INTERP_PAIR,
                       f"Autoencoder reconstruction, off-grid pair {INTERP_PAIR} (position I3, never seen)")
        context_history(iex, icontact, ires, INTERP_SUB)
        methods_comparison(iex, icontact, ires, INTERP_SUB, name="03_methods_comparison.png")
    print("DONE", FINAL, flush=True)


if __name__ == "__main__":
    main()
