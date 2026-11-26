#!/usr/bin/env python3
"""Final 162-trajectory CESN pipeline. Run from repository root in maniskill-esn."""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'CLEAVER_OPTIMIZED'
DATA=ROOT/'datasets/cleaver_extended_workspace_162_complete/cleaver_same_goal_162.h5'
OUT=BASE/'outputs/final162'
FIG=ROOT/'paper_figures/final162'
# Zero-based object IDs: K2 validation, K5/K6 test. All lie in train hull.
GROUPS={'train':[0,2,3,6,7,8], 'validation':[1], 'test':[4,5]}
FRAMES=[1,2,3,5,10,12,23,45,67,112,157,223]
PARAMS=dict(reservoir=1500,radius=.6,leak=.015,gain=1.,ridge=10.,seed=42)

def patch():
    changes={}
    p=BASE/'config.py'; s=p.read_text()
    s=s.replace('"cleaver_extended_workspace_162"','"cleaver_extended_workspace_162_complete"')
    s=s.replace('OUT = BASE / "outputs"\n','OUT = BASE / "outputs" / "final162"\n')
    for k,v in [('SPECTRAL_RADIUS','0.6'),('LEAK_RATE','0.015'),('RIDGE','10.0')]:
        s=re.sub(rf'(?m)^{k}\s*=.*$',f'{k} = {v}',s)
    s=re.sub(r'(?m)^AE_DIR\s*=.*$', 'AE_DIR = OUT / "autoencoder_lightweight128_matched"',s)
    s=re.sub(r'(?m)^LATENT\s*=.*$', 'LATENT = AE_DIR / "cleaver_lightweight_latent128.h5"',s)
    if 'VALIDATION = set(SPLIT["validation"])' not in s:
        s=s.replace('TEST = set(SPLIT["test"])','TEST = set(SPLIT["test"])\nVALIDATION = set(SPLIT["validation"])')
    assert 'cleaver_extended_workspace_162_complete' in s and '"final162"' in s
    changes[p]=s
    p=BASE/'autoencoder/train_variable.py'; s=p.read_text()
    if '# FINAL162_FIXED_SPLIT' not in s:
        old='assert set(pair)==set(cfg.TRAIN)|set(cfg.TEST)'
        assert s.count(old)==1, 'Unexpected train_variable load_data source'
        s=s.replace(old,'assert set(pair)==set(cfg.TRAIN)|set(cfg.SPLIT["validation"])|set(cfg.TEST)')
        start=s.index('def split_pairs('); end=s.index('def device_rgb(',start)
        s=s[:start]+'''# FINAL162_FIXED_SPLIT

def split_pairs(pair,obj):
    return tuple(sorted(map(int,cfg.SPLIT[k])) for k in ('train','validation','test'))

'''+s[end:]
        start=s.index('    original=cfg.OUT/"autoencoder_lightweight128"');end=s.index('    ids=',start)
        s=s[:start]+s[end:]
        start=s.index('    source=torch.load(original/');end=s.index('    if args.resume:',start)
        s=s[:start]+'''    signature.update(dynamic_dim=args.dynamic_dim,static_training='from_scratch')
'''+s[end:]
    assert 'source=torch.load(original/' not in s and 'saved_split==' not in s
    changes[p]=s
    p=BASE/'cesn/model.py'; s=p.read_text()
    old='assert len(rows)==126 and list(map(len,sets))==[48,6,9]'
    new='assert len(rows)==2*sum(map(len,sets)) and all(sets)'
    if old in s: s=s.replace(old,new)
    assert new in s, 'Unexpected CESN loader source'
    changes[p]=s
    p=BASE/'analysis/visualize_dataset_positions.py';s=p.read_text().replace('datasets/cleaver_extended_workspace_162/','datasets/cleaver_extended_workspace_162_complete/').replace('paper_figures/extended_workspace','paper_figures/final162')
    changes[p]=s
    for p,s in changes.items(): compile(s,str(p),'exec')
    for p,s in changes.items(): p.write_text(s)
    split={k:[o*9+r for o in objs for r in range(9)] for k,objs in GROUPS.items()}
    split['dataset']=str(DATA.relative_to(ROOT))
    (BASE/'split.json').write_text(json.dumps(split,indent=2)+'\n')
    return split

def audit():
    import h5py, numpy as np
    from scipy.spatial import Delaunay
    coords={}
    with h5py.File(DATA,'r') as f:
        keys={k for k in f if k.startswith('trajectory_')}
        assert keys=={f'trajectory_{i:03d}' for i in range(162)}
        for p in range(81):
            a,b=[f[f'trajectory_{2*p+l:03d}'] for l in (0,1)]
            for l,g in enumerate((a,b)):
                assert (int(g.attrs['pair_id']),int(g.attrs['object_id']),int(g.attrs['start_id']),int(g.attrs['label']))==(p,p//9,p%9,l)
                assert g['rgb'].shape==(223,256,256,3) and g['rgb'].dtype==np.uint8
                for name in ('qpos','target_world','knife_xy','contact_impulses'):
                    assert np.isfinite(g[name][:]).all(),(p,l,name)
                hit=np.flatnonzero(g['contact_impulses'][:,:2].sum(1)>1e-4)
                assert bool(len(hit))==bool(l),(p,l,'contact')
                xy=g['target_world'][:2]
                if p//9 in coords: assert np.allclose(coords[p//9],xy,atol=1e-7,rtol=0)
                coords[p//9]=xy
            assert np.allclose(a['qpos'][0],b['qpos'][0],atol=1e-5,rtol=0)
            assert np.allclose(a['target_world'][:],b['target_world'][:],atol=1e-7,rtol=0)
    hull=Delaunay(np.array([coords[o] for o in GROUPS['train']],dtype=float))
    for o in GROUPS['validation']+GROUPS['test']:
        assert hull.find_simplex(coords[o],tol=1e-6)>=0, ('outside train hull',o,coords[o])
    print('AUDIT PASSED: 162 trajectories, paired contact labels, held-out goals inside training hull',flush=True)

def run(*args):
    print('RUN:',*args,flush=True);subprocess.run([sys.executable,'-u',*map(str,args)],cwd=ROOT,check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--batch-size',type=int,default=750);ap.add_argument('--validation-only',action='store_true');a=ap.parse_args()
    os.chdir(ROOT);os.environ.setdefault('MPLBACKEND','Agg')
    import torch
    assert torch.cuda.is_available(),'CUDA required; activate maniskill-esn'
    assert DATA.is_file(),DATA
    if OUT.exists() and not a.resume: raise RuntimeError('Final output already exists: use --resume to continue')
    audit();split=patch();OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    run('-m','CLEAVER_OPTIMIZED.analysis.visualize_dataset_positions','--dataset',DATA,'--out',FIG)
    ae=OUT/'autoencoder_lightweight128_matched';lat=ae/'cleaver_lightweight_latent128.h5'
    if not all(p.is_file() for p in (ae/'training_summary.json',lat,ae/'autoencoder.pt',ae/'split.json',ae/'autoencoder_safe.png',ae/'autoencoder_pain.png')):
        cmd=['-m','CLEAVER_OPTIMIZED.autoencoder.train_variable','--dynamic-dim','128','--static-epochs','15','--joint-epochs','20','--patience','20','--batch-size',str(a.batch_size)]
        if (ae/'latest.pt').exists(): cmd+=['--resume']
        run(*cmd)
    import shutil, numpy as np
    from types import SimpleNamespace
    from CLEAVER_OPTIMIZED.cesn import model as b
    from CLEAVER_OPTIMIZED.experiments.run_dimensions import normalize,rgb_metrics
    from CLEAVER_OPTIMIZED.experiments.run_forward_lowleak import make_model
    for name in ('autoencoder_safe.png','autoencoder_pain.png'): shutil.copy2(ae/name,FIG/name)
    data,used=b.load(lat,ae/'split.json');assert all(used[k]==split[k] for k in GROUPS)
    assert all(d['z'].shape[-1]==128 for d in data.values())
    device=torch.device('cuda');torch.manual_seed(42);torch.set_float32_matmul_precision('high')
    info,lo,hi=normalize(data,device);scale=info['dynamic_scale'];out=OUT/'CESN_Vision';out.mkdir(exist_ok=True)
    sig=dict(parameters=PARAMS,split=used,latent_sha256=b.digest(lat),frames=FRAMES,restarts=9,max_iter=40,trajectory_batch=3,core_sha256=b.digest(Path(b.__file__)),runner_sha256=b.digest(Path(__file__)))
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==sig,'Run identity changed; do not reuse cached metrics'
    else:b.atomic_json(mp,sig)
    b.atomic_json(out/'normalization.json',info);b.atomic_json(out/'split_used.json',used)
    weights=b.common_weights(1500,42,1.,device);model=make_model(weights,1500,.6,.015,1.,42,device)
    b.sync(device);tick=time.perf_counter();model.fit(data['train'],10.);b.sync(device);seconds=time.perf_counter()-tick
    b.atomic_pt(out/'cesn.pt',dict(W=model.W.cpu(),Wc=model.Wc.cpu(),Wf=model.Wf.cpu(),bias=model.bias.cpu(),readout=model.readout.cpu(),normalization=info,split=used,leak=.015,parameters=PARAMS))
    starts=torch.tensor(np.random.default_rng(42).uniform(-.95,.95,(9,2)),device=device,dtype=torch.float32);starts[0]=0
    opts=SimpleNamespace(max_iter=40); summaries=[];oracles=[];rgbs=[]
    for sn in (['validation'] if a.validation_only else ['validation','test']):
        d=data[sn];part=out/sn;part.mkdir(exist_ok=True);o=model.oracle(d,1.0);oracles.append(dict(split=sn,train_seconds=seconds,**o));print('ORACLE',sn,o,flush=True)
        rows=[]
        for k in FRAMES:
            for i in range(0,len(d['keys']),3):
                fp=part/f'prefix_{k:03d}_{i:03d}.json';ids=list(range(i,min(i+3,len(d['keys']))))
                if fp.exists():rr=json.loads(fp.read_text())
                else:rr=b.invert(model,d,ids,k,starts,opts,lo,hi,scale);b.atomic_json(fp,rr)
                rows.extend(rr)
            rr=[r for r in rows if r['frames']==k];print(sn,k,'accuracy',100*np.mean([r['correct'] for r in rr]),'XY mm',np.mean([r['xy_error_mm'] for r in rr]),flush=True)
        b.csv_write(part/'per_trajectory.csv',rows)
        for r in b.summarize(rows):summaries.append(dict(split=sn,**r,future_nrmse=None if r['future_rmse'] is None else r['future_rmse']/scale))
        rgbs.extend(dict(split=sn,**r) for r in rgb_metrics(model,d,info,ae/'autoencoder.pt',128,device))
        b.csv_write(out/'comparison.csv',summaries);b.csv_write(out/'oracle_forward.csv',oracles);b.csv_write(out/'rgb_sample_metrics.csv',rgbs)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for sn in sorted({r['split'] for r in summaries}):
        for metric,label,filename in [('accuracy_pct','Intent accuracy (%)','01_intent_accuracy'),('xy_mean_mm','XY error (mm)','02_xy_error'),('future_nrmse','Future latent NRMSE','03_future_rmse'),('amortized_inference_ms','Inversion time (ms/trajectory)','04_inference_time'),('pain_recall_pct','Precontact PAIN recall (%)','05_pain_recall'),('mean_margin','Signed classification margin','06_classification_margin')]:
            subset='precontact' if metric=='pain_recall_pct' else 'all';rr=[r for r in summaries if r['split']==sn and r['subset']==subset and r[metric] is not None]
            if not rr:continue
            fig,ax=plt.subplots(figsize=(3.5,2.5));ax.plot([r['frames'] for r in rr],[r[metric] for r in rr],marker='o',ms=3);ax.set(xlabel='Observed frames',ylabel=label,xscale='log');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIG/f'{sn}_{filename}.png',dpi=300);fig.savefig(FIG/f'{sn}_{filename}.pdf');plt.close(fig)
    selected=dict(model_name='CESN-Vision',source_configuration='C161',dataset=str(DATA.relative_to(ROOT)),static_latent_dim=128,dynamic_latent_dim=128,reservoir_size=1500,spectral_radius=.6,leak_rate=.015,ridge=10.,feedback_gain=1.,seed=42,prediction_mode='absolute',autoencoder_directory=str(ae.relative_to(BASE)),checkpoint=str((out/'cesn.pt').relative_to(BASE)),split_file='split.json')
    b.atomic_json(BASE/'selected_model.json',selected)
    # Preserve small scientific outputs outside ignored model directories.
    paper=ROOT/'paper_results'/'final162';paper.mkdir(parents=True,exist_ok=True)
    for src in (out/'comparison.csv',out/'oracle_forward.csv',out/'rgb_sample_metrics.csv',out/'normalization.json',out/'split_used.json',out/'manifest.json',ae/'training_summary.json'):
        shutil.copy2(src,paper/src.name)
    for sn in (['validation'] if a.validation_only else ['validation','test']):
        shutil.copy2(out/sn/'per_trajectory.csv',paper/f'{sn}_per_trajectory.csv')
    print('FINAL162 COMPLETE:',out, 'FIGURES:',FIG, 'PAPER RESULTS:',paper,flush=True)

if __name__=='__main__':main()
