"""Run from repository root: python audit_cleaver.py. Read-only models/data."""
import csv, json, hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import h5py
import torch
from CLEAVER import config as cfg
from CLEAVER.analysis import invert_explicit as inv
from CLEAVER.core import train_explicit as tr
from CLEAVER.core import train_autoencoder as ae_train
from CLEAVER.core import extract_latents as ext

OUT = cfg.OUT / 'audit'
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN, TEST = set(cfg.TRAIN), set(cfg.TEST)
assert len(TRAIN)==70 and len(TEST)==11 and not TRAIN & TEST
assert TRAIN | TEST == set(range(81))

def load(p):
    return torch.load(p, map_location='cpu', weights_only=False)

def check_split(c, train_key, name):
    assert set(map(int,c[train_key])) == TRAIN, name+' train mismatch'
    assert set(map(int,c['test_pairs'])) == TEST, name+' test mismatch'
    print(name, 'checkpoint split OK')

def write(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

ac=load(cfg.AE_MODEL); ec=load(cfg.EXPLICIT_DIR/'cesn.pt'); sc=load(cfg.SPATIAL_DIR/'cesn.pt')
check_split(ac,'final_train_pairs','AE')
check_split(ec,'train_pairs','Explicit')
check_split(sc,'train_pairs','Spatial')
assert set(ae_train.build_final_splits(cfg.RGB)['train70']) == TRAIN
assert set(tr.build_pair_splits()['train']) == TRAIN
print('PCA recomputation using training pairs only')
raw_train=tr.load_raw_sequences(TRAIN)
pca=tr.fit_static_pca(raw_train)
pm=ec['pca_mean'].numpy(); pc=ec['pca_components'].numpy()
a=np.asarray(pca['components']); b=pc
# Compare subspaces, insensitive to component sign.
if a.shape[0]!=16: a=a.T
if b.shape[0]!=16: b=b.T
print('PCA mean max difference:', np.max(np.abs(pca['mean']-pm)))
print('PCA subspace max difference:',np.max(np.abs(a.T@a-b.T@b)))
assert np.allclose(pca['mean'],pm,atol=1e-5,rtol=1e-5)
assert np.allclose(a.T@a,b.T@b,atol=1e-4,rtol=1e-4)
assert np.allclose(sc['pca_mean'].numpy(),pm,atol=1e-5)
sp=sc['pca_components'].numpy(); sp=sp if sp.shape[0]==16 else sp.T
assert np.allclose(sp.T@sp,b.T@b,atol=1e-4)
print('PCA verified (mean and subspace)')

model=inv.ContextualESNTorch.from_state_dict(ec['model_state'],device=DEVICE)
encoder=ext.PainStaticDynamicAutoencoderV6Native256Latent200(static_latent_dim=128,dynamic_latent_dim=200).to(DEVICE)
encoder.load_state_dict(ac['model_state_dict']); encoder.eval()

def encode_prefix(rgb):
    with torch.inference_mode(), torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=='cuda'):
        f=torch.stack([ext.prepare_rgb_native256(x,DEVICE) for x in rgb]); z=encoder.encode_static(f[:1]); d=encoder.encode_dynamic(f-f[:1])
    return z[0].float().cpu().numpy(),d.float().cpu().numpy()

pairs=defaultdict(dict); hashes=defaultdict(set)
with h5py.File(cfg.RGB,'r') as h,h5py.File(cfg.LATENT,'r') as l:
    for k in h:
        if not k.startswith('trajectory_'): continue
        g=h[k]; p=int(g.attrs['pair_id']); y=int(g.attrs['label']); pairs[p][y]=k
        assert int(l[k].attrs['pair_id'])==p and int(l[k].attrs['pain_label'])==y
        hashes[hashlib.sha256(g['rgb'][0].tobytes()).hexdigest()].add(p)
    assert set(pairs)==TRAIN|TEST
    assert all(set(v)=={0,1} for v in pairs.values())
    cross=[sorted(v) for v in hashes.values() if v&TRAIN and v&TEST]
    print('Exact initial RGB duplicates across split:',cross)
    counts=defaultdict(lambda: [0,0]); diagnostics=[]; items=[]
    for p,keys in sorted(pairs.items()):
        g0,g1=[h[keys[y]] for y in (0,1)]
        obj=int(g0.attrs['object_id']); counts[obj][int(p in TEST)]+=1
        assert np.allclose(g0['target_world'][:],g1['target_world'][:],atol=1e-7,rtol=0)
        if p not in TEST: continue
        rgb=[g['rgb'][:12] for g in (g0,g1)]
        zs=[]; zd=[]
        for y in (0,1):
            z,d=encode_prefix(rgb[y]); zs.append(z); zd.append(d)
            lg=l[keys[y]]; cached=lg['z_dynamic'][:12]; stored=lg['z_static'][:]
            print('Prefix check',keys[y], 'static max',float(np.max(np.abs(z-stored))), 'dynamic max',float(np.max(np.abs(d-cached))))
            # FP16 convolution algorithms can differ by batch size. Report differences, use freshly encoded data.
        hits=np.flatnonzero(np.asarray(g1['contact_impulses'])[:,:2].sum(1)>1e-4); assert len(hits)
        contact=int(hits[0]); goal=g0['target_world'][:2]
        for n in range(1,13):
            diff=np.abs(rgb[0][n-1].astype(float)-rgb[1][n-1].astype(float))
            diagnostics.append(dict(pair=p,frame=n-1,rgb_mae_255=float(diff.mean()),rgb_max_255=float(diff.max()),static_l2=float(np.linalg.norm(zs[0]-zs[1])),dynamic_l2=float(np.linalg.norm(zd[0][n-1]-zd[1][n-1])),first_contact=contact))
        for y in (0,1): items.append((p,y,zs[y],zd[y],goal,contact))
    print('Object coverage (train pairs, test pairs):',dict(counts))
    print('Unseen test objects:',[k for k,(a,b) in counts.items() if a==0 and b>0])
write('paired_visual_diagnostics.csv',diagnostics)

# Compare original grid to independent random restarts, identical for both classes.
original=inv.task_restart_points
rng=np.random.default_rng(2026)
random_starts=list(map(tuple,rng.uniform(-.85,.85,(9,2))*[inv.GRASP_X_SCALE,inv.GRASP_Y_SCALE]))
rows=[]
for mode in ('grid','random'):
    inv.task_restart_points=original if mode=='grid' else lambda label: random_starts
    for n in (1,2,3,5,10,12):
        sub=[]
        for p,y,z,d,goal,contact in items:
            ps=inv.transform_static(z,pm,pc)
            sol=[inv.optimize_xy_batched(model,d[:n],ps,c,80,20) for c in (0,1)]
            losses=[v[2] for v in sol]; pred=int(losses[1]<losses[0]); correct=losses[y]; wrong=losses[1-y]
            margin=(wrong-correct)/max(wrong+correct,1e-12)
            row=dict(mode=mode,frames=n,pair=p,label=y,pred=pred,correct=int(pred==y),safe_loss=losses[0],pain_loss=losses[1],signed_margin=margin,xy_error_mm=float(np.linalg.norm(np.array(sol[pred][:2])-goal)*1000),precontact=int(n<=contact),first_contact=contact)
            rows.append(row); sub.append(row)
        print(mode,n,'frames | accuracy',np.mean([r['correct'] for r in sub]),'median margin',np.median([r['signed_margin'] for r in sub]),'precontact',sum(r['precontact'] for r in sub),flush=True)
    write('early_inversion.csv',rows)
inv.task_restart_points=original
# Exact identical observation control: both labels get SAME static and dynamic input.
control=[]
for p,y,z,d,goal,contact in items:
    if y!=0: continue
    ps=inv.transform_static(z,pm,pc)
    sol=[inv.optimize_xy_batched(model,d[:1],ps,c,80,20) for c in (0,1)]
    pred=int(sol[1][2]<sol[0][2]); control.extend(dict(pair=p,label=k,pred=pred,correct=int(k==pred)) for k in (0,1))
write('identical_input_control.csv',control)
print('Identical-input control accuracy:',np.mean([r['correct'] for r in control]))
print('Audit files:',OUT)
