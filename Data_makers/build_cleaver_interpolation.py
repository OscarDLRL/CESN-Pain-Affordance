"""Run in maniskill-data. Reuse the final local generator without modifying it."""
import argparse, json, sys
from pathlib import Path
import h5py
import numpy as np
from CLEAVER.config import RGB, TRAIN, ROOT
from Data_makers import build_cleaver_dataset_162_fixed as gen


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    out=ROOT/'datasets/cleaver_interpolation_162'
    plan_path=out.parent/'cleaver_interpolation_plan.json'
    with h5py.File(RGB,'r') as h:
        train=np.array([g['target_world'][:2] for k,g in h.items() if k.startswith('trajectory_') and int(g.attrs['pair_id']) in TRAIN])
        starts={int(g.attrs['start_id']):np.asarray(g['requested_start_xyz'][:]) for k,g in h.items() if k.startswith('trajectory_')}
    lo=train.min(0); hi=train.max(0)
    # Predeclared interior fractions; independent of optimizer/grid resolution.
    fractions=np.array([(x,y) for y in (.21,.53,.78) for x in (.18,.47,.81)])
    goals=lo+(hi-lo)*fractions
    from scipy.spatial import Delaunay
    assert np.all(Delaunay(np.unique(train,axis=0)).find_simplex(goals)>=0)
    assert np.min(np.linalg.norm(goals[:,None]-train[None],axis=2))>1e-3
    assert set(starts)==set(range(9))
    assert np.allclose(np.array([starts[i] for i in range(9)])[:,2],gen.START_CENTER[2])
    gen.START_OFFSETS=[tuple(starts[i][:2]-gen.START_CENTER[:2]) for i in range(9)]
    gen.OBJECT_OFFSETS=[tuple(g-gen.v4.GOAL[:2]) for g in goals]
    plan={'goals_m':goals.tolist(),'starts_m':[starts[i].tolist() for i in range(9)],'train_bounds_m':[lo.tolist(),hi.tolist()]}
    if out.exists():
        if not a.resume: raise RuntimeError('Output exists; use --resume')
        assert json.loads(plan_path.read_text())==plan, 'Interpolation plan changed'
    else:
        plan_path.write_text(json.dumps(plan,indent=2))
    print('Interpolation goals (mm):\n',goals*1000,flush=True)
    sys.argv=[sys.argv[0],'--out',str(out)]
    if a.resume: sys.argv+=['--resume']
    if a.pilot: sys.argv+=['--pairs']+[str(9*i+4) for i in range(9)]
    gen.main()

if __name__=='__main__': main()
