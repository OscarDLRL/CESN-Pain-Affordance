"""Export actual initial simulator RGB frames and optionally restore an accepted scene."""
import argparse,csv
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def read(path):
    rows={}; labels={}
    with h5py.File(path,'r') as h:
        for key in sorted(h):
            if not key.startswith('trajectory_'): continue
            g=h[key]; p=int(g.attrs['pair_id']); label=int(g.attrs['label']); o=int(g.attrs['object_id']); s=int(g.attrs['start_id'])
            if label in labels.setdefault(p,set()): raise ValueError(f'Duplicate label: {p}')
            labels[p].add(label)
            if label!=0: continue
            rows[o,s]=dict(pair=p,key=key,rgb=g['rgb'][0],knife=np.asarray(g['knife_xy'][:],float),tcp=np.asarray(g['prepared_tcp'][:],float),requested=np.asarray(g['requested_start_xyz'][:],float))
        if len(rows)!=81 or len(labels)!=81 or any(x!={0,1} for x in labels.values()): raise ValueError('Expected 81 complete accepted pairs in extended dataset')
    return rows

def figures(rows,out,show):
    import matplotlib
    if not show: matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    def grid(fixed,robot):
        fig,axes=plt.subplots(3,3,figsize=(7.1,7.0),layout='constrained')
        for i,ax in enumerate(axes.flat):
            row=rows.get((fixed,i) if robot else (i,fixed)); ax.set_xticks([]);ax.set_yticks([])
            if row is None: ax.text(.5,.5,'Not accepted',ha='center',va='center',transform=ax.transAxes,fontsize=10);ax.set_title(f'{"R" if robot else "K"}{i+1}');continue
            ax.imshow(row['rgb']); xy=(row['requested'] if robot else row['knife'])[:2]*100
            ax.set_title(f'{"R" if robot else "K"}{i+1}: ({xy[0]:.1f}, {xy[1]:.1f}) cm',fontsize=9)
        name='01_robot_initial_rgb' if robot else '02_knife_positions_rgb'
        fig.suptitle(f'Fixed cleaver K{fixed+1}' if robot else f'Fixed robot start R{fixed+1}',fontsize=12)
        fig.savefig(out/f'{name}.png',dpi=300)
        if not show: plt.close(fig)
    # Select one fixed configuration with maximal accepted coverage; tie favors central index.
    obj=max(range(9),key=lambda o:(sum((o,s) in rows for s in range(9)),-abs(o-4),-o))
    start=max(range(9),key=lambda s:(sum((o,s) in rows for o in range(9)),-abs(s-4),-s))
    grid(obj,True);grid(start,False)
    fig,ax=plt.subplots(figsize=(4.8,4.4),layout='constrained'); matrix=np.array([[(o,s) in rows for s in range(9)] for o in range(9)])
    ax.imshow(matrix,cmap='Greys',vmin=0,vmax=1);ax.set(xticks=range(9),yticks=range(9),xticklabels=[f'R{i+1}' for i in range(9)],yticklabels=[f'K{i+1}' for i in range(9)],xlabel='Robot initial position',ylabel='Cleaver position')
    for o in range(9):
        for s in range(9): ax.text(s,o,'1' if matrix[o,s] else '0',ha='center',va='center',color='white' if matrix[o,s] else 'black',fontsize=8)
    fig.savefig(out/'03_accepted_pair_coverage.png',dpi=300)
    if not show: plt.close(fig)
    with (out/'dataset_positions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['pair_id','trajectory','object_id','start_id','knife_x_m','knife_y_m','requested_tcp_x_m','requested_tcp_y_m','requested_tcp_z_m','actual_tcp_x_m','actual_tcp_y_m','actual_tcp_z_m'])
        for (o,s),r in sorted(rows.items()):w.writerow([r['pair'],r['key'],o,s,*r['knife'][:2],*r['requested'][:3],*r['tcp'][:3]])
    print(f'Accepted pairs: {len(rows)}; fixed cleaver K{obj+1}; fixed robot R{start+1}; figures: {out}')
    if show:plt.show()

def viewer(path,rows,pair):
    import gymnasium as gym
    import sapien
    from Data_makers import test_cleaver_dynamic_grasp_v4 as v4
    row=next((r for r in rows.values() if r['pair']==pair),None)
    if row is None: raise ValueError(f'Pair {pair} is not accepted')
    with h5py.File(path,'r') as h:
        g=h[row['key']]; q=np.asarray(g['prepared_qpos'][:],np.float32).reshape(1,-1); pos=np.asarray(g['object_xyz'][0],float).reshape(3); quat=np.asarray(g['object_quat'][0],float).reshape(4)
    env=gym.make(v4.ENV_ID,obs_mode='rgb',control_mode='pd_joint_pos',render_mode='human',sim_backend='physx_cpu',num_envs=1,max_episode_steps=1500)
    try:
        env.reset(seed=123); e=env.unwrapped
        e.agent.robot.set_qpos(q); e.agent.robot.set_qvel(np.zeros_like(q)); e.obj.set_pose(sapien.Pose(p=pos,q=quat)); e.obj.set_linear_velocity(np.zeros(3,np.float32)); e.obj.set_angular_velocity(np.zeros(3,np.float32))
        print(f'Restored accepted pair {pair}. Close viewer to exit. No physics steps or IK are executed.')
        v=e.render_human()
        while not v.closed: e.render_human()
    finally:env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'datasets/cleaver_extended_workspace_162_complete/cleaver_same_goal_162.h5');p.add_argument('--out',type=Path,default=ROOT/'paper_figures/final162');p.add_argument('--show',action='store_true');p.add_argument('--viewer',action='store_true');p.add_argument('--pair',type=int,default=40);a=p.parse_args(); rows=read(a.dataset)
    if a.viewer:viewer(a.dataset,rows,a.pair)
    else:figures(rows,a.out,a.show)
if __name__=='__main__':main()
