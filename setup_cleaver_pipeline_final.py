#!/usr/bin/env python3
"""Create an isolated cleaver training pipeline from the saved v1 experiment.
Run from the repository root with maniskill-esn. Does not train or edit legacy files.
"""
import ast
import json
import re
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
import h5py
import numpy as np

ROOT = Path.cwd().resolve()
TAG = 'v1.0-pain-affordance'
RAW = ROOT / 'datasets/cleaver_same_goal_162_final/cleaver_same_goal_162.h5'
PKG = ROOT / 'CLEAVER'
LAT = ROOT / 'datasets/cleaver_latent200/cleaver_latent200.h5'
if PKG.exists():
    raise SystemExit('CLEAVER already exists; refusing to overwrite. Run existing stages instead.')
if LAT.exists():
    raise SystemExit('Latent file already exists; refusing to overwrite.')
if not RAW.is_file():
    raise SystemExit(f'Missing dataset: {RAW}')

pairs = defaultdict(dict)
with h5py.File(RAW, 'r') as h:
    for name in sorted(h):
        if not name.startswith('trajectory_'): continue
        g=h[name]; pid=int(g.attrs['pair_id']); label=int(g.attrs['label'])
        if label not in (0,1) or label in pairs[pid]: raise ValueError(f'Invalid pair {pid}')
        if g['rgb'].shape[1:] != (256,256,3) or g['rgb'].dtype != np.uint8: raise ValueError(name)
        if len(g['rgb'])!=223: raise ValueError(f'Expected 223 frames: {name}')
        target=np.asarray(g['target_world'][:],float)
        if target.shape!=(3,) or not np.isfinite(target).all(): raise ValueError(name)
        edge=np.asarray(g['contact_impulses'])[:,:2].sum(axis=1)
        hits=np.flatnonzero(edge>1e-4)
        if (label==0 and len(hits)) or (label==1 and not len(hits)): raise ValueError(f'Contact mismatch {name}')
        pairs[pid][label]={'key':name,'goal':target.tolist(),'frames':len(g['rgb']),'object':int(g.attrs['object_id']),'start':int(g.attrs['start_id']),'hit':int(hits[0]) if len(hits) else None}
    for pid,p in pairs.items():
        if set(p)!={0,1}: raise ValueError(f'Incomplete pair {pid}')
        a,b=p[0],p[1]
        if a['frames']!=b['frames'] or not np.allclose(a['goal'],b['goal'],atol=1e-7,rtol=0): raise ValueError(f'Pair mismatch {pid}')
        if (a['object'],a['start'])!=divmod(pid,9) or a['object']!=b['object'] or a['start']!=b['start']: raise ValueError(f'Pair IDs mismatch {pid}')
        g0,g1=h[a['key']],h[b['key']]
        if np.max(np.abs(g0['qpos'][0]-g1['qpos'][0]))>1e-5: raise ValueError(f'Initial qpos mismatch {pid}')
        if np.mean(np.abs(g0['rgb'][0].astype(float)-g1['rgb'][0].astype(float)))>.25: raise ValueError(f'Initial RGB mismatch {pid}')
        if b['hit']>int(g1['close_end'][()])+1: raise ValueError(f'Late edge contact {pid}')
if set(pairs)!=set(range(81)): raise SystemExit(f'Expected all 81 accepted pairs, found {len(pairs)}; do not silently change split.')
ids=np.array(sorted(pairs),dtype=int); np.random.default_rng(42).shuffle(ids)
train=sorted(ids[:70].tolist()); test=sorted(ids[70:].tolist())
# Restart grid is the public experimental design, independent of test labels and observations.
restarts=[(x,y) for y in (-.045,-.015,.015) for x in (.02,.05,.08)]


def get(path):
    return subprocess.check_output(['git','show',f'{TAG}:{path}'],text=True)

def replace_func(src,name,new):
    node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=src.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+new.rstrip()+'\n'+''.join(lines[node.end_lineno:])

def replace_once(src,pattern,repl):
    out,n=re.subn(pattern,repl,src,count=1,flags=re.S)
    if n!=1: raise RuntimeError(f'Expected source pattern not found: {pattern}')
    return out

def inject(src,code):
    marker='if __name__ == "__main__":'
    if marker not in src: raise RuntimeError('Missing main guard')
    return src.replace(marker,code+'\n\n'+marker,1)

config='''from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
RGB = ROOT / "datasets/cleaver_same_goal_162_final/cleaver_same_goal_162.h5"
LATENT = ROOT / "datasets/cleaver_latent200/cleaver_latent200.h5"
OUT = ROOT / "CLEAVER/outputs"
AE_DIR = OUT / "autoencoder"
AE_MODEL = AE_DIR / "autoencoder.pt"
EXPLICIT_DIR = OUT / "explicit"
SPATIAL_DIR = OUT / "spatial_only"
SPLIT = json.loads((ROOT / "CLEAVER/split.json").read_text())
TRAIN = set(SPLIT["train"])
TEST = set(SPLIT["test"])
RESTARTS = '''+repr(restarts)+'\n'

sources={}
# Autoencoder: same network/losses; same data split for all models.
s=get('CESN/core/train_autoencoder_v6_native256_latent200_final70.py')
s=replace_func(s,'build_final_splits','''def build_final_splits(path):
    from CLEAVER.config import TRAIN, TEST
    return {"train70": TRAIN, "test11": TEST}
''')
sources['core/train_autoencoder.py']=inject(s,'''from CLEAVER.config import RGB, AE_DIR, AE_MODEL
DATA_PATH = RGB
OUT_DIR = AE_DIR
MODEL_PATH = AE_MODEL
SUMMARY_PATH = OUT_DIR / "training_summary.json"''')
s=get('CESN/core/extract_latents_v6_native256_latent200_final70.py')
sources['core/extract_latents.py']=inject(s,'''from CLEAVER.config import RGB, AE_MODEL, LATENT
RGB_DATA = RGB
MODEL_PATH = AE_MODEL
OUT_DIR = LATENT.parent
OUT_H5 = LATENT''')

s=get('CESN/core/train_grasp_context_v6_native256_latent200_ridge05_final70.py')
s=replace_func(s,'build_pair_splits','''def build_pair_splits():
    from CLEAVER.config import TRAIN, TEST
    return {"train": TRAIN, "validation": set(), "test": TEST}
''')
s=replace_once(s,r'grasp_x = \(\s*knife_x\s*\+\s*\(\s*SAFE_TARGET_LOCAL_X\s*if label == 0\s*else PAIN_TARGET_LOCAL_X\s*\)\s*\)\s*grasp_y = \(\s*knife_y\s*\)', 'grasp_x, grasp_y = map(float, rg["target_world"][:2])')
sources['core/train_explicit.py']=inject(s,'''from CLEAVER.config import RGB, LATENT, EXPLICIT_DIR
RGB_DATA_PATH = RGB
LATENT_DATA_PATH = LATENT
OUT_DIR = EXPLICIT_DIR
MODEL_PATH = OUT_DIR / "cesn.pt"
TRAIN_PAIRS = 70
VAL_PAIRS = 0''')

s=get('CESN_spatial_only/core/train_spatial_only.py')
s=replace_func(s,'build_pair_splits','''def build_pair_splits():
    from CLEAVER.config import TRAIN, TEST
    return {"train": TRAIN, "test": TEST}
''')
s=replace_once(s,r'grasp_x = knife_x \+ \(\s*SAFE_TARGET_LOCAL_X if label == 0 else PAIN_TARGET_LOCAL_X\s*\)\s*grasp_y = knife_y','grasp_x, grasp_y = map(float, rg["target_world"][:2])')
sources['core/train_spatial.py']=inject(s,'''from CLEAVER.config import RGB, LATENT, SPATIAL_DIR
RGB_DATA_PATH = RGB
LATENT_DATA_PATH = LATENT
OUT_DIR = SPATIAL_DIR
MODEL_PATH = OUT_DIR / "cesn.pt"
TRAIN_PAIRS = 70
TEST_PAIRS = 11''')

s=get('CESN/analysis/method_a_lbfgs_v6_native256_latent200_ridge05_final70_gpu.py')
s=replace_func(s,'task_restart_points','''def task_restart_points(label):
    from CLEAVER.config import RESTARTS
    return RESTARTS
''')
s=replace_once(s,r'true_x = knife_x \+ \(SAFE_TARGET_LOCAL_X if label == 0 else PAIN_TARGET_LOCAL_X\)\s*true_y = knife_y','true_x, true_y = map(float, rgb_h5[key]["target_world"][:2])')
# Both members of a pair use the same pre-contact eligibility criterion.
s=s.replace('"true_grasp_y": float(true_y),','"true_grasp_y": float(true_y),\n                "pair_contact_frame": int(np.flatnonzero(np.asarray(rgb_h5[f"trajectory_{pair_id*2+1:03d}"]["contact_impulses"])[:,:2].sum(axis=1)>1e-4)[0]),',1)
s=s.replace('joint_task_correct = []','joint_task_correct = []\n    precontact_correct = []\n    precontact_xy = []',1)
s=s.replace('joint_task_correct.append(joint_ok)','joint_task_correct.append(joint_ok)\n        if n_obs <= item["pair_contact_frame"]:\n            precontact_correct.append(joint_ok)\n            precontact_xy.append(xy_error_mm(joint_x, joint_y, true_x, true_y))',1)
s=s.replace('"history_percent": int(pct),','"history_percent": int(pct),\n        "precontact_n": len(precontact_correct),\n        "precontact_task_accuracy": float(np.mean(precontact_correct)) if precontact_correct else None,\n        "precontact_xy_mean_mm": float(np.mean(precontact_xy)) if precontact_xy else None,',1)
sources['analysis/invert_explicit.py']=inject(s,'''from CLEAVER.config import RGB, LATENT, EXPLICIT_DIR
RGB_DATA_PATH = RGB
LATENT_DATA_PATH = LATENT
OUT_DIR = EXPLICIT_DIR
MODEL_PATH = OUT_DIR / "cesn.pt"
RESULT_DIR = OUT_DIR / "inversion"
SUMMARY_CSV = RESULT_DIR / "summary.csv"
SUMMARY_JSON = RESULT_DIR / "summary.json"''')

s=get('CESN_spatial_only/analysis/invert_spatial_only.py')
s=replace_once(s,r'true_x = knife_x \+ \(\s*SAFE_TARGET_LOCAL_X if label == 0 else PAIN_TARGET_LOCAL_X\s*\)\s*true_y = knife_y','true_x, true_y = map(float, rg["target_world"][:2])')
s=s.replace('pred_label = 0 if relative_x >= 0.0 else 1','pred_label = 0  # constant-class baseline; spatial-only has no intent output')
s=s.replace('Infer only grasp XY, then derive affordance from geometry.','Infer grasp XY only. Intent accuracy is a constant SAFE baseline, not a CESN prediction.').replace('AFFORDANCE=','CONSTANT BASELINE=').replace('"affordance_accuracy_pct"','"constant_safe_baseline_accuracy_pct"').replace('relative_grasp_x >= 0 -> SAFE; < 0 -> PAIN','constant SAFE baseline; no intent output from this model')
sources['analysis/invert_spatial.py']=inject(s,'''from CLEAVER.config import RGB, LATENT, SPATIAL_DIR, RESTARTS
RGB_DATA_PATH = RGB
LATENT_DATA_PATH = LATENT
OUT_DIR = SPATIAL_DIR
MODEL_PATH = OUT_DIR / "cesn.pt"
RESTARTS_XY = RESTARTS''')

sources['analysis/plot_results.py']='''from CLEAVER.config import EXPLICIT_DIR, SPATIAL_DIR, OUT
import pandas as pd
import matplotlib.pyplot as plt

a=pd.read_csv(EXPLICIT_DIR/'inversion/summary.csv')
b=pd.read_csv(SPATIAL_DIR/'inversion/spatial_only_summary.csv')
out=OUT/'figures'; out.mkdir(exist_ok=True)
fig,ax=plt.subplots(); ax.plot(a.history_percent,100*a.joint_task_accuracy,'o-',label='Explicit CESN'); ax.plot(a.history_percent,100*a.precontact_task_accuracy,'s--',label='Pre-contact subset'); ax.axhline(50,ls=':',label='Balanced constant baseline'); ax.set(xlabel='Observed trajectory (%)',ylabel='Intent accuracy (%)',ylim=(0,105)); ax.legend(); fig.tight_layout(); fig.savefig(out/'intent_accuracy.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(); ax.plot(a.history_percent,a.joint_xy_mean_mm,'o-',label='Explicit'); ax.plot(b.history_percent,b.xy_mean_mm,'s-',label='Spatial only'); ax.set(xlabel='Observed trajectory (%)',ylabel='Mean XY error (mm)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'xy_error.png',dpi=180); plt.close(fig)
'''
# Compile all files before writing anything.
for name,src in sources.items(): compile(src,name,'exec')
compile(config,'config.py','exec')
PKG.mkdir()
for d in ('core','analysis'): (PKG/d).mkdir(); (PKG/d/'__init__.py').touch()
(PKG/'__init__.py').touch()
(PKG/'config.py').write_text(config)
(PKG/'split.json').write_text(json.dumps({'seed':42,'train':train,'test':test,'pair_count':len(pairs),'object_counts':dict(Counter(p[0]['object'] for p in pairs.values())),'pairs':pairs},indent=2)+'\n')
for name,src in sources.items(): (PKG/name).write_text(src)
print('Created',PKG,'with train/test counts',len(train),len(test))
print('Accepted pairs per object:',dict(Counter(p[0]['object'] for p in pairs.values())))
