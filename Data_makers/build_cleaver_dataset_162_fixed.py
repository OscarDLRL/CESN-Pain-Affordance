#!/usr/bin/env python3
"""Batch generator using the successful dynamic-grasp v4 pilot unchanged.
81 paired scenes (9 object positions x 9 initial TCP offsets). Resume with --resume.
Only accepted complete pairs enter HDF5. Run in maniskill-data from project root.
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import h5py
import numpy as np
from Data_makers import test_cleaver_dynamic_grasp_v4 as v4

# Conservative offsets around the successful pilot (meters).
OBJECT_OFFSETS = [(x,y) for y in (.03,0.,-.03) for x in (-.06,-.03,0.)]
START_CENTER = np.array([-.05,-.05,.35])
START_OFFSETS = [(x,y) for y in (.04,0.,-.04) for x in (-.04,0.,.04)]

def validate(a,b,args):
    s=v4.base.initial_pair_check(a,b)
    s.update(safe=v4.summarize(a,args),hazard=v4.summarize(b,args))
    s['same_frame_count']=len(a['rgb'])==len(b['rgb'])
    s['final_reach_pair_error_m']=float(np.linalg.norm(a['tcp_xyz'][int(a['reach_end'])]-b['tcp_xyz'][int(b['reach_end'])]))
    s['accepted']=bool(s['same_frame_count'] and s['initial_rgb_mae']<=.25 and s['initial_qpos_max_abs']<=1e-5 and s['initial_object_xyz_max_abs']<=1e-6 and s['final_reach_pair_error_m']<=.005 and all(s[k]['goal_error_m']<=.005 and s[k]['lift_success'] for k in ('safe','hazard')) and s['safe']['first_edge_contact_frame'] is None and s['hazard']['first_edge_contact_frame'] is not None and s['hazard']['first_edge_contact_frame']<=s['hazard']['close_end']+1)
    return s

def write_group(g,ep,pair_id,label,oi,si,object_xy,start_offset):
    g.attrs.update(pair_id=pair_id,trajectory_id=pair_id*2+label,label=label,variant=('safe','hazard')[label],object_id=oi,start_id=si,frames=len(ep['rgb']))
    for k,val in ep.items():
        arr=np.asarray(val)
        if arr.dtype.kind not in 'biuf': continue
        g.create_dataset(k,data=arr,**({'compression':'gzip','compression_opts':4} if arr.ndim and arr.size>32 else {}))
    g.create_dataset('target_world',data=v4.GOAL)
    g.create_dataset('target_local',data=np.array([-v4.LENGTH/2+.025,0.,0.]))
    g.create_dataset('knife_xy',data=object_xy)
    g.create_dataset('start_offset_xy',data=start_offset)
    g.create_dataset('prepared_qpos',data=ep['qpos'][0])
    g.create_dataset('prepared_tcp',data=ep['tcp_xyz'][0])
    g.create_dataset('requested_start_xyz',data=v4.COMMON_START)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=Path('datasets/cleaver_same_goal_162_fixed')); p.add_argument('--pairs',nargs='+',type=int); p.add_argument('--resume',action='store_true'); args=p.parse_args()
    ids=list(range(81)) if args.pairs is None else list(dict.fromkeys(args.pairs))
    if any(i<0 or i>80 for i in ids): p.error('pair IDs must be 0..80')
    args.out.mkdir(parents=True,exist_ok=args.resume)
    reports=args.out/'pairs'; reports.mkdir(exist_ok=True)
    path=args.out/'cleaver_same_goal_162.h5'
    cfg=SimpleNamespace(steps=100,knots=100,settle=12,close_steps=30,lift_steps=60,hold_steps=20,lift=.05,standoff=.07,force_threshold=.1,impulse_threshold=.0001,balance_weight=.5,out=None)
    origin=v4.ORIGIN.copy(); goal=v4.GOAL.copy(); start=START_CENTER.copy()
    with h5py.File(path,'a' if args.resume else 'x') as h:
        if args.resume and len(h) and h.attrs.get('start_policy','')!='fixed_world_grid_v1':
            raise RuntimeError('Cannot resume a dataset with a different start policy')
        h.attrs['start_policy']='fixed_world_grid_v1'
        h.attrs['start_center_xyz']=START_CENTER
        h.attrs.update(dataset_name='cleaver_same_goal_162',primary_observation='RGB 256x256',requested_trajectories=162,label_0='safe',label_1='hazard',object_locations=9,robot_starts=9,variants=2,fixture_note='Virtual pose reset until close; unassisted dynamic lift; contact sampled at control boundaries.')
        # Remove incomplete transactions from interrupted runs.
        for k in list(h.keys()):
            if k.startswith('_pending_'): del h[k]
        for pid in ids:
            names=[f'trajectory_{pid*2+l:03d}' for l in (0,1)]
            if all(n in h for n in names): print('Already saved',pid,flush=True); continue
            for n in names:
                if n in h: del h[n]
            oi,si=divmod(pid,9); ox,oy=OBJECT_OFFSETS[oi]; sx,sy=START_OFFSETS[si]
            delta=np.array([ox,oy,0.]); v4.ORIGIN=origin+delta; v4.GOAL=goal+delta
            # Fixed world-frame grid; no local search or class-dependent start.
            v4.COMMON_START=start+np.array([sx,sy,0.])
            cfg.out=reports/f'pair_{pid:03d}'; cfg.out.mkdir(exist_ok=True)
            row=dict(pair_id=pid,object_id=oi,start_id=si,object_offset=[ox,oy],start_offset=[sx,sy],requested_start_xyz_m=v4.COMMON_START.tolist(),accepted=False)
            env=None
            try:
                raw=gym.make(v4.ENV_ID,obs_mode='rgb',control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='physx_cpu',num_envs=1,max_episode_steps=1500)
                env=v4.Capture(raw)
                pa,prep=v4.run(env,cfg,'safe',start_candidate=v4.COMMON_START,plan_only=True)
                pb,_=v4.run(env,cfg,'hazard',prep=prep,plan_only=True)
                if np.max(np.abs(pa['initial_qpos']-pb['initial_qpos']))>1e-5:
                    raise RuntimeError('Pair initial joint states differ')
                a,_=v4.run(env,cfg,'safe',prep=prep,saved_plan=pa)
                b,_=v4.run(env,cfg,'hazard',prep=prep,saved_plan=pb)
                row.update(validate(a,b,cfg)); row['goal_xyz_m']=v4.GOAL.tolist()
                row['requested_start_xyz_m']=v4.COMMON_START.tolist()
                row['actual_start_xyz_m']=a['tcp_xyz'][0].tolist()
                row['start_error_m']=float(np.linalg.norm(a['tcp_xyz'][0]-v4.COMMON_START))
                row['accepted']=bool(row['accepted'] and row['start_error_m']<=.005)
                if row['accepted']:
                    for label,ep in enumerate((a,b)):
                        g=h.create_group('_pending_'+names[label]); write_group(g,ep,pid,label,oi,si,v4.ORIGIN[:2],np.array([sx,sy]))
                    for n in names: h.move('_pending_'+n,n)
                    h.flush()
                    if pid==40 or not (args.out/'example_safe.gif').exists():
                        v4.base.save_gif(a['rgb'],args.out/'example_safe.gif'); v4.base.save_gif(b['rgb'],args.out/'example_hazard.gif')
            except RuntimeError as exc:
                row.update(accepted=False,error=str(exc))
            finally:
                if env is not None: env.close()
            (cfg.out/'summary.json').write_text(json.dumps(row,indent=2)+'\n')
            print(f'PAIR {pid:02d}: accepted={row["accepted"]} {row.get("error", "")}',flush=True)
        # Build manifest from actual HDF5 contents, not attempted trajectories.
        rows=[]
        for name in sorted(h):
            if not name.startswith('trajectory_'): continue
            g=h[name]; rows.append({k:g.attrs[k] for k in ('trajectory_id','pair_id','object_id','start_id','variant','label','frames')})
        with (args.out/'manifest.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['trajectory_id','pair_id','object_id','start_id','variant','label','frames']); w.writeheader(); w.writerows(rows)
        attempted=len(list(reports.glob('pair_*/summary.json')))
        summary=dict(accepted_pairs=len(rows)//2,accepted_trajectories=len(rows),attempted_pairs=attempted,requested_pairs=81,failed_pairs=attempted-len(rows)//2,h5=str(path))
        h.attrs.update(accepted_pairs=len(rows)//2,failed_pairs=summary['failed_pairs'])
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
