"""Validation-only discrete inference; historical filename retained for compatibility."""
import argparse
import numpy as np
import torch
from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn import model as cesn
from CLEAVER_OPTIMIZED.experiments.forward_sweep_cv import pooled_folds,build
from CLEAVER_OPTIMIZED.experiments.discrete_inference import history,perceived_xy,fixed_losses
from CLEAVER_OPTIMIZED.perception import GoalPerception,template_from,collect,fit_h
import h5py

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ae-run',required=True); a=ap.parse_args()
    dev=torch.device('cuda'); folder=cfg.OUT/'autoencoder'/a.ae_run; out=folder/'discrete_cv'; out.mkdir(exist_ok=True)
    rows=[]; alarms=[]
    for obj,d in pooled_folds(folder/'latents.h5'):
        info,lo,hi=cesn.normalize(d,dev); train=d['train']; val=d['validation']; keys=set(train['keys'])
        with h5py.File(cfg.RGB,'r') as f:
            rgb=f[train['keys'][0]]['rgb']; tmpl=template_from(rgb[0],rgb[cfg.REFERENCE_FRAME])
        raw=collect(cfg.RGB,tmpl)
        perception=GoalPerception(tmpl,fit_h([r for r in raw if r['key'] in keys])); xy,_=perceived_xy(val,perception,lo,hi)
        model=build(dev,1,cfg.FEEDBACK_GAIN); model.fit(train,cfg.RIDGE)
        for k in (1,5,10):
            L=history(model,val,xy,k).cpu().numpy(); labs=val['label']
            for n in cfg.INVERSION_FRAMES:
                rows.append(dict(object=obj,obs_every=k,frames=n,accuracy_pct=100*float(np.mean(L[:,n-1].argmin(1)==labs))))
            steps=np.arange(0,L.shape[1],k); score=(L[:,steps,0]-L[:,steps,1])/(L[:,steps].sum(2)+1e-12)
            for tau in (0,.05,.1,.15,.2,.3):
                hits=[]
                for i in range(len(labs)):
                    run=0; alarm=None
                    for t,v in zip(steps,score[i]):
                        run=run+1 if v>tau else 0
                        if run>=cfg.DETECT_CONSECUTIVE: alarm=int(t); break
                    hits.append(alarm)
                pain=[i for i in range(len(labs)) if labs[i]==1]; safe=[i for i in range(len(labs)) if labs[i]==0]
                lead=[int(val['contact'][i])-hits[i] for i in pain if hits[i] is not None and hits[i]<val['contact'][i]]
                alarms.append(dict(object=obj,obs_every=k,tau=tau,detected_pct=100*len(lead)/len(pain),false_alarm_pct=100*sum(hits[i] is not None for i in safe)/len(safe),median_lead=float(np.median(lead)) if lead else None))
        print('fold',obj,flush=True)
    cesn.csv_write(out/'classification.csv',rows); cesn.csv_write(out/'online_fixed.csv',alarms)
if __name__=='__main__': main()
