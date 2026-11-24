"""Qualitative 9-joint completion using frozen CESN and selected Method A/L-BFGS.

Observed prefix is optimized; predictions for K:T are evaluated only afterward.
Joint 0..6 units radians, joint 7..8 finger positions meters.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from CESN.core import cesn
from CESN.analysis.final.experiment_common import (
    OUTPUT,load_environment,xy_of,invert_group,K_from_fraction,save_csv,
    output_dir,save_metadata)

DEFAULT_CASES=['pick_interpolation_0080','pick_extrapolation_0120']

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',default='cuda');ap.add_argument('--dtype',choices=('float32','float64'),default='float32')
    ap.add_argument('--trajectory',action='append',default=None)
    ap.add_argument('--fraction',type=float,default=.45)
    ap.add_argument('--restarts',type=int,default=4)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--out-dir',type=Path,default=OUTPUT/'movement_completion')
    ap.add_argument('--overwrite',action='store_true')
    ap.add_argument('--task-space',action='store_true',help='Also render Panda TCP task-space using ManiSkill FK (optional dependency)')
    a=ap.parse_args()
    if not 0<a.fraction<1:ap.error('must observe a proper prefix: 0<fraction<1')
    model,runner,test,train,bounds=load_environment(a.device,a.dtype)
    ids=a.trajectory or DEFAULT_CASES
    chosen=[]
    for tr in ids:
        matches=[r for r in test if str(r['id'])==tr]
        if len(matches)!=1:raise ValueError(f'Expected one trajectory {tr}, found {len(matches)}')
        chosen.append(matches[0])
    out=output_dir(a.out_dir,a.overwrite);(out/'figures').mkdir(exist_ok=True)
    scores=[]
    fk=None
    if a.task_space:
        from CESN.analysis.legacy_adam.reporting_spatial_final import PandaTCPForwardKinematics
        fk=PandaTCPForwardKinematics()
    try:
        for i,row in enumerate(chosen):
            true=np.asarray(row['joints'],dtype=np.float64);T=len(true);K=K_from_fraction(T,a.fraction)
            result,stats=invert_group(runner,[row],[K],bounds,a.seed+i,a.restarts)
            inv=result[0]; context=cesn.make_context(inv.predicted_task,inv.estimated_xy)
            pred=np.asarray(cesn.predict_trajectory(model,context,T),dtype=np.float64)
            error=np.sqrt(np.mean((pred[K:]-true[K:])**2,axis=0))
            fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True)
            for j,ax in enumerate(axes.flat):
                unit='rad' if j<7 else 'm'
                ax.plot(np.arange(T),true[:,j],label='Observed demonstration')
                ax.plot(np.arange(T),pred[:,j],'--',label='Recovered context, forward CESN')
                ax.axvline(K-.5,color='grey',linestyle=':')
                ax.set(title=f'Joint {j}: future RMSE {error[j]:.5f} {unit}',ylabel=unit)
                ax.grid(alpha=.2)
            for ax in axes[-1]:ax.set_xlabel('Frame')
            h,l=axes.flat[0].get_legend_handles_labels()
            fig.legend(h,l,loc='upper center',ncol=2,bbox_to_anchor=(.5,.99))
            fig.suptitle(f'{row["id"]} | {K}/{T} observed | task {row["kind"]} → {inv.predicted_task}',y=1.03)
            fig.tight_layout(rect=(0,0,1,.95))
            name=f'09_joint_completion_{row["id"]}_K{K}.png'
            fig.savefig(out/'figures'/name,dpi=190,bbox_inches='tight');plt.close(fig)
            scores.append(dict(trajectory=str(row['id']),region=row.get('context_region','unknown'),
                       fraction=K/T,T=T,K=K,true_task=str(row['kind']).lower(),pred_task=inv.predicted_task,
                       task_correct=int(str(row['kind']).lower()==inv.predicted_task),
                       xy_error_mm=1000*np.linalg.norm(inv.estimated_xy-xy_of(row)),
                       future_rmse=float(np.sqrt(np.mean((pred[K:]-true[K:])**2))),
                       **{f'joint_{j}_future_rmse':float(error[j]) for j in range(9)}))
            print('completion',row['id'],'task',inv.predicted_task,'XY mm',scores[-1]['xy_error_mm'])
            if fk is not None:
                # Plot the original Adam-era *visual protocol*, but with
                # the current selected Method A + L-BFGS estimated context.
                true_tcp = fk(true)
                recovered_tcp = fk(pred)
                true_xy = xy_of(row)
                xy_error_mm = 1000.0 * np.linalg.norm(inv.estimated_xy - true_xy)

                # Representative alternative spatial contexts from TRAIN only.
                # No future ground truth is used to select these contexts.
                train_xy = np.stack([xy_of(r) for r in train])
                centre = train_xy.mean(axis=0)
                angles = np.arctan2(train_xy[:, 1] - centre[1],
                                    train_xy[:, 0] - centre[0])
                ordered = train_xy[np.argsort(angles)]
                indices = np.rint(np.linspace(0, len(ordered) - 1,
                                             min(8, len(ordered)))).astype(int)
                alternatives = []
                for xy in ordered[indices]:
                    if np.linalg.norm(xy - true_xy) < .005:
                        continue
                    if np.linalg.norm(xy - inv.estimated_xy) < .005:
                        continue
                    alt_ctx = cesn.make_context(inv.predicted_task, xy)
                    alt_q = np.asarray(cesn.predict_trajectory(model, alt_ctx, T),
                                       dtype=np.float64)
                    alternatives.append(fk(alt_q))

                fig, ax = plt.subplots(figsize=(8.4, 6.2))
                for j, tcp in enumerate(alternatives):
                    ax.plot(tcp[:, 0], tcp[:, 2], lw=1.1, alpha=.13,
                            label='Alternative CESN contexts' if j == 0 else None,
                            zorder=1)
                ax.plot(recovered_tcp[:, 0], recovered_tcp[:, 2],
                        color='#a4ad13', lw=2.7,
                        label='CESN (L-BFGS recovered context)', zorder=3)
                ax.plot(true_tcp[K-1:, 0], true_tcp[K-1:, 2],
                        color='#17bac5', ls=':', lw=2.7,
                        label='True hidden future', zorder=4)
                ax.plot(true_tcp[:K, 0], true_tcp[:K, 2],
                        color='#1f77b4', marker='o', markersize=4, lw=1.9,
                        label=f'Observed prefix ({K/T:.0%})', zorder=5)
                ax.scatter(true_tcp[K-1, 0], true_tcp[K-1, 2],
                           s=78, color='#1f77b4', zorder=6,
                           label='Last observed frame')
                cube = np.asarray(row['cube_xyz'], dtype=np.float64)
                cube0 = cube[0] if cube.ndim == 2 else cube
                ax.scatter(cube0[0], cube0[2], color='#ff7f0e',
                           marker='s', s=85, zorder=7, label='True cube')

                fig.suptitle('Task-space context inversion — '
                             f'{str(row["kind"]).capitalize()}, '
                             f'{row.get("context_region", "unknown")}',
                             fontsize=15, y=.985)
                fig.text(.5, .936,
                         f'{K/T:.0%} observed  |  task: '
                         f'{str(row["kind"]).lower()} → {inv.predicted_task}'
                         f'  |  context XY error: {xy_error_mm:.1f} mm',
                         ha='center', va='center', fontsize=10.5)
                ax.set(xlabel='TCP x [m]', ylabel='TCP z [m]')
                ax.grid(alpha=.25)
                ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.025),
                          ncol=2, frameon=False, fontsize=8.5)
                fig.tight_layout(rect=(.04, .04, .98, .82))
                fig.savefig(out/'figures'/f'10_task_space_{row["id"]}_K{K}.png',
                            dpi=220, bbox_inches='tight')
                plt.close(fig)
    finally:
        if fk is not None:fk.close()
    save_csv(out/'movement_completion_detail.csv',scores)
    save_metadata(out,experiment='movement_completion',fraction=a.fraction,
                  trajectory_ids=ids,seed=a.seed,restarts=a.restarts,
                  note='Task-space generated only when --task-space is specified and ManiSkill FK works.')
    print('Saved:',out)
if __name__=='__main__':main()
