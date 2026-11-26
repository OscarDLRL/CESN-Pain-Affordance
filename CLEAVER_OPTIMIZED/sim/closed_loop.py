"""Closed-loop test in ManiSkill: the robot executes its plan, the online pain
detector watches the camera, and on alarm the robot STOPS (holds its current joint
configuration and opens the gripper).

Scene, plans, virtual fixture and contact sensing are the dataset generator's own
code (Data_makers); only the execution loop is replaced. Test locations (objects
4 and 5) were never used to fit the AE, the CESN, the perception or the threshold.

Per trajectory: alarm frame, nominal edge-contact frame (from the dataset), edge
contact in the closed-loop run (impulse on the blade edge), goal-XY error of the
perception, detector latency, and how closely the replay matches the dataset
frames before the stop.

    python -m CLEAVER_OPTIMIZED.sim.closed_loop                     # 18 held-out grid test pairs
    python -m CLEAVER_OPTIMIZED.sim.closed_loop --protocol offgrid  # 24 off-grid pairs (interpolation)
    python -m CLEAVER_OPTIMIZED.sim.closed_loop --pairs 36 45       # subset
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import h5py
import numpy as np

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn.model import csv_write
from CLEAVER_OPTIMIZED.detector import OnlinePainDetector
from Data_makers import build_cleaver_dataset_162_fixed as gen

v4 = gen.v4
base = v4.base
ORIGIN0, GOAL0 = v4.ORIGIN.copy(), v4.GOAL.copy()
GEN = SimpleNamespace(steps=100, knots=100, settle=12, close_steps=30, lift_steps=60, hold_steps=20, lift=.05,
                      standoff=.07, force_threshold=.1, impulse_threshold=.0001, balance_weight=.5, out=None)
LABELS = ("safe", "hazard")


def configure_pair(pid, workspace):
    """Grid dataset: offsets from workspace_config.json. Off-grid: absolute positions from plan.json."""
    oi, si = divmod(pid, 9)
    if "objects_m" in workspace:
        delta = np.r_[np.asarray(workspace["objects_m"][oi]) - ORIGIN0[:2], 0.0]
        v4.COMMON_START = np.asarray(workspace["starts_m"][si], float)
    else:
        ox, oy = workspace["object_offsets"][oi]
        sx, sy = workspace["start_offsets"][si]
        delta = np.array([ox, oy, 0.0])
        v4.COMMON_START = np.array(workspace["start_center"]) + np.array([sx, sy, 0.0])
    v4.ORIGIN, v4.GOAL = ORIGIN0 + delta, GOAL0 + delta
    return oi, si


def edge_contact_frame(contact_impulses):
    c = np.asarray(contact_impulses)
    hits = np.flatnonzero(c[:, 0] + c[:, 1] > GEN.impulse_threshold)
    return int(hits[0]) if len(hits) else None


def run_closed_loop(env, label, prep, plan, detector):
    """Execute one planned trajectory with the detector in the loop (mirrors v4.run)."""
    env.reset(seed=123)
    e = env.unwrapped
    planner = base.new_planner(env, vis=False)
    try:
        for a in prep:
            env.step(a)
        open_cmd = float(prep[-1, -1])
        e.reset_fixture()
        env.restart_capture_from_current_state()
        if np.max(np.abs(base.get_qpos(env) - plan["initial_qpos"])) > 1e-5:
            raise RuntimeError("prepared state differs from the planned state")
        cmds = plan["commands"]
        schedule = ([np.r_[q, open_cmd] for q in cmds[1:]] + [np.r_[cmds[-1], open_cmd]] * GEN.settle
                    + [np.r_[cmds[-1], -1.0]] * GEN.close_steps)
        hold = None
        lift = None
        detector.alarm_step, detector.history = None, []  # nothing carried over between episodes

        def after_step():
            f = len(env.rgb) - 1
            if f == cfg.REFERENCE_FRAME:
                detector.reset(env.rgb[f])
            if f >= cfg.REFERENCE_FRAME and detector.alarm_step is None:
                detector.step(env.rgb[f] if detector.needs_frame() else None)

        while len(env.rgb) < cfg.NUM_FRAMES:
            if hold is None and detector.alarm_step is not None:
                # STOP: keep the current joint configuration, open the gripper
                hold = np.r_[base.get_qpos(env)[:len(cmds[0])], open_cmd].astype(np.float32)
            if hold is not None:
                action = hold
            elif schedule:
                action = schedule.pop(0).astype(np.float32)
            else:
                if lift is None:  # nominal lift, planned from the actual TCP (as in v4.run)
                    pos = base.get_tcp_xyz(env).astype(float)
                    quat = np.asarray(e.agent.tcp.pose.sp.q, dtype=float)
                    pts = np.stack([pos, pos + [0, 0, GEN.lift / 2], pos + [0, 0, GEN.lift]])
                    lift = [np.r_[q, -1.0] for q in v4.cartesian_plan(planner, env, pts, [quat] * 3, GEN.lift_steps, GEN.knots)[1:]]
                    lift += [lift[-1]] * GEN.hold_steps
                    env.fixture = False
                action = lift.pop(0).astype(np.float32) if lift else np.r_[base.get_qpos(env)[:len(cmds[0])], -1.0].astype(np.float32)
            env.step(action)
            after_step()
        return env.episode()
    finally:
        planner.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["heldout_grid", "offgrid"], default="heldout_grid")
    ap.add_argument("--pairs", nargs="+", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gifs", type=int, default=2, help="save GIFs for the first N pairs")
    a = ap.parse_args()
    offgrid = a.protocol == "offgrid"
    source = cfg.OFFGRID if offgrid else cfg.RGB
    out = Path(a.out or cfg.OUT / ("sim_closed_loop_offgrid" if offgrid else "sim_closed_loop"))
    out.mkdir(parents=True, exist_ok=True)
    workspace = json.loads((cfg.OFFGRID_PLAN if offgrid else cfg.RGB.parent / "workspace_config.json").read_text())
    ds = h5py.File(source, "r")
    if a.pairs is None:
        a.pairs = sorted({int(ds[k].attrs["pair_id"]) for k in ds if k.startswith("trajectory_")}) if offgrid else cfg.TEST
    detector = OnlinePainDetector.build(protocol=a.protocol)
    rows = []
    for n_pair, pid in enumerate(a.pairs):
        oi, si = configure_pair(pid, workspace)
        t0 = time.perf_counter()
        env = None
        try:
            env = v4.Capture(gym.make(v4.ENV_ID, obs_mode="rgb", control_mode="pd_joint_pos", render_mode="rgb_array",
                                      sim_backend="physx_cpu", num_envs=1, max_episode_steps=1500))
            plans = {}
            plans["safe"], prep = v4.run(env, GEN, "safe", start_candidate=v4.COMMON_START, plan_only=True)
            plans["hazard"], _ = v4.run(env, GEN, "hazard", prep=prep, plan_only=True)
            for label_id, label in enumerate(LABELS):
                ep = run_closed_loop(env, label, prep, plans[label], detector)
                key = f"trajectory_{2 * pid + label_id:03d}"
                nominal = edge_contact_frame(ds[key]["contact_impulses"][:]) if key in ds else None
                closed = edge_contact_frame(ep["contact_impulses"])
                alarm_frame = None if detector.alarm_step is None else detector.alarm_step + cfg.REFERENCE_FRAME
                upto = alarm_frame if alarm_frame is not None else cfg.NUM_FRAMES
                replay_mae = float(np.abs(np.asarray(ep["rgb"][:upto], np.float32) - ds[key]["rgb"][:upto]).mean() / 255) if key in ds else None
                goal_true = np.asarray(v4.GOAL[:2])
                obs_ms = [h["ms"] for h in detector.history if h["observed"]]
                row = dict(pair=pid, object=oi, start=si, label=label_id, alarm_frame=alarm_frame,
                           nominal_edge_contact_frame=nominal, closed_loop_edge_contact_frame=closed,
                           stopped_before_contact=int(alarm_frame is not None and nominal is not None and alarm_frame < nominal),
                           edge_contact_avoided=int(label_id == 1 and closed is None),
                           false_stop=int(label_id == 0 and alarm_frame is not None),
                           lead_frames=(nominal - alarm_frame) if (alarm_frame is not None and nominal is not None) else None,
                           goal_xy_error_mm=float(np.linalg.norm(detector.goal_xy - goal_true) * 1000),
                           match_score=detector.match_score, ms_per_step=float(np.mean([h["ms"] for h in detector.history])),
                           ms_per_observation=float(np.mean(obs_ms)) if obs_ms else None,
                           replay_rgb_mae_before_stop=replay_mae)
                rows.append(row)
                print(f"pair {pid} {label:6s} alarm={alarm_frame} nominal_contact={nominal} closed_contact={closed} "
                      f"goal_err={row['goal_xy_error_mm']:.1f}mm replay_mae={replay_mae}", flush=True)
                if n_pair < a.gifs:
                    base.save_gif(ep["rgb"], out / f"pair{pid:03d}_{label}.gif")
        except RuntimeError as exc:
            rows.append(dict(pair=pid, object=oi, start=si, label=-1, error=str(exc)))
            print(f"pair {pid} FAILED: {exc}", flush=True)
        finally:
            if env is not None:
                env.close()
        csv_write(out / "per_trajectory.csv", [r for r in rows if r.get("label", -1) >= 0])
        print(f"pair {pid} done in {time.perf_counter() - t0:.0f}s", flush=True)

    ok = [r for r in rows if r.get("label", -1) >= 0]
    pain = [r for r in ok if r["label"] == 1]
    safe = [r for r in ok if r["label"] == 0]
    summary = dict(
        pairs=len(a.pairs), failed_pairs=len([r for r in rows if r.get("label", -1) < 0]),
        pain_trajectories=len(pain), safe_trajectories=len(safe),
        pain_stopped_before_contact_pct=100 * np.mean([r["stopped_before_contact"] for r in pain]) if pain else None,
        pain_edge_contact_avoided_pct=100 * np.mean([r["edge_contact_avoided"] for r in pain]) if pain else None,
        safe_false_stop_pct=100 * np.mean([r["false_stop"] for r in safe]) if safe else None,
        median_lead_frames=float(np.median([r["lead_frames"] for r in pain if r["lead_frames"] is not None])) if pain else None,
        min_lead_frames=min([r["lead_frames"] for r in pain if r["lead_frames"] is not None], default=None),
        goal_xy_error_mm_mean=float(np.mean([r["goal_xy_error_mm"] for r in ok])) if ok else None,
        ms_per_step=float(np.mean([r["ms_per_step"] for r in ok])) if ok else None,
        ms_per_observation=float(np.mean([r["ms_per_observation"] for r in ok])) if ok else None,
        protocol=a.protocol, settings=dict(obs_every=cfg.OBS_EVERY, tau=cfg.DETECT_TAU, consecutive=cfg.DETECT_CONSECUTIVE, ae_run=cfg.AE_RUN))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
