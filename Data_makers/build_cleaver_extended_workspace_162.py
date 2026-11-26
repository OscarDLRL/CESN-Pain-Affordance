#!/usr/bin/env python3
"""Complete the extended workspace dataset without changing the validated physics."""
import argparse
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from Data_makers import build_cleaver_dataset_162_fixed as generator

START_CENTER = np.array([-.05, -.05, .35])
START_OFFSETS = [(x,y) for y in (.08,0.,-.08) for x in (-.08,0.,.08)]
ORIGINAL_OFFSETS = [(x,y) for y in (.06,0.,-.06) for x in (-.09,-.03,.03)]
PILOT = list(range(45,54)) + list(range(72,81))
DEFAULT_OUT = Path('datasets/cleaver_extended_workspace_162_complete')


def inspect_dataset(out, expected):
    path = out/'cleaver_same_goal_162.h5'
    valid, bad = set(), []
    with h5py.File(path,'r') as h:
        for pid in range(81):
            names = [f'trajectory_{2*pid+l:03d}' for l in (0,1)]
            if not any(n in h for n in names):
                continue
            if not all(n in h for n in names):
                bad.append((pid,'incomplete pair')); continue
            oi,si = divmod(pid,9)
            a,b = (h[n] for n in names)
            for l,g in enumerate((a,b)):
                if (int(g.attrs['pair_id']) != pid or int(g.attrs['object_id']) != oi
                    or int(g.attrs['start_id']) != si or int(g.attrs['label']) != l
                    or g['rgb'].shape != (223,256,256,3) or g['rgb'].dtype != np.uint8):
                    bad.append((pid,'metadata or RGB mismatch')); break
                if not np.allclose(g['knife_xy'][...], expected[oi], atol=1e-6, rtol=0):
                    bad.append((pid,'object position mismatch')); break
            else:
                if not np.allclose(a['target_world'][...],b['target_world'][...],atol=1e-6,rtol=0):
                    bad.append((pid,'different paired goals')); continue
                report = json.loads((out/'pairs'/f'pair_{pid:03d}'/'summary.json').read_text())
                if not report.get('accepted'):
                    bad.append((pid,'not accepted by original validator')); continue
                valid.add(pid)
    if bad:
        raise RuntimeError(f'Dataset integrity errors: {bad}')
    return valid


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--k6-x',type=float,default=0.,help='K6 X offset in meters relative to original cleaver origin')
    p.add_argument('--k9-x',type=float,default=0.,help='K9 X offset in meters relative to original cleaver origin')
    p.add_argument('--k9-y',type=float,default=-.03,help='K9 Y offset in meters relative to original cleaver origin')
    p.add_argument('--pilot',action='store_true',help='Test all 18 pairs involving replacement positions')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--show-config',action='store_true')
    args=p.parse_args()
    offsets=ORIGINAL_OFFSETS.copy()
    offsets[5]=(args.k6_x,0.)
    offsets[8]=(args.k9_x,args.k9_y)
    if len(set(offsets))!=9 or not all(np.isfinite(offsets).flat):
        p.error('All nine positions must be finite and distinct')
    if any(not (-.09 <= x <= .03 and -.06 <= y <= .06) for x,y in offsets):
        p.error('Replacement offsets must remain within the original workspace bounds')
    out=args.out.resolve()
    if out == Path('datasets/cleaver_extended_workspace_162').resolve():
        p.error('The existing 126-trajectory dataset must not be overwritten')
    expected=[(generator.v4.ORIGIN+np.array([x,y,0.]))[:2].tolist() for x,y in offsets]
    spec={'start_center':START_CENTER.tolist(),'start_offsets':[list(x) for x in START_OFFSETS],
          'object_offsets':[list(x) for x in offsets],'object_xy_world':expected}
    print('Robot positions (m):')
    for i,(x,y) in enumerate(START_OFFSETS): print(f'R{i+1}: {START_CENTER+np.array([x,y,0.])}')
    print('Cleaver positions (m):')
    for i,xy in enumerate(expected): print(f'K{i+1}: {xy}')
    print('Output:',out,flush=True)
    if args.show_config: return
    marker=out/'workspace_config.json'
    if args.resume:
        if not marker.is_file():
            p.error('Missing workspace configuration')
        previous=json.loads(marker.read_text())
        if previous != spec:
            # Permit a K9-only correction if no accepted K9 trajectory exists.
            if not args.pilot or any(previous.get(k)!=spec[k] for k in ('start_center','start_offsets')) or any(previous[k][:8]!=spec[k][:8] for k in ('object_offsets','object_xy_world')):
                p.error('Only K9 may change during a resumed pilot')
            with h5py.File(out/'cleaver_same_goal_162.h5','r') as h:
                if any(f'trajectory_{2*pid+l:03d}' in h for pid in range(72,81) for l in (0,1)):
                    p.error('K9 already has saved trajectories; cannot change its coordinates')
            inspect_dataset(out,expected)
            tmp=marker.with_suffix('.tmp')
            tmp.write_text(json.dumps(spec,indent=2)+'\n')
            tmp.replace(marker)
            print('Updated K9 coordinates; preserved all accepted pairs.',flush=True)
        valid=inspect_dataset(out,expected)
        if not args.pilot and not set(PILOT).issubset(valid):
            p.error('All 18 replacement pairs must pass the pilot before completing the dataset')
    elif out.exists():
        p.error('Output already exists; use --resume with identical coordinates or choose a new --out')
    elif not args.pilot:
        p.error('Start with --pilot, then use --resume to complete all 81 pairs')
    else:
        # The original generator permits an existing output directory only in resume mode.
        out.mkdir(parents=True,exist_ok=False)
        marker.write_text(json.dumps(spec,indent=2)+'\n')
    generator.START_CENTER=START_CENTER.copy()
    generator.START_OFFSETS=START_OFFSETS
    generator.OBJECT_OFFSETS=offsets
    old_argv=sys.argv[:]
    try:
        sys.argv=[old_argv[0],'--out',str(out),'--resume']
        if args.pilot: sys.argv+=['--pairs',*map(str,PILOT)]
        generator.main()
    finally:
        sys.argv=old_argv
    valid=inspect_dataset(out,expected)
    required=set(PILOT if args.pilot else range(81))
    missing=sorted(required-valid)
    print(f'VALIDATED: {len(required & valid)}/{len(required)} requested pairs')
    if missing:
        print('FAILED OR MISSING PAIRS:',missing)
        raise SystemExit(2)
    print('PILOT PASSED: 18/18 pairs' if args.pilot else 'DATASET COMPLETE: 81 pairs / 162 trajectories',flush=True)

if __name__=='__main__': main()
