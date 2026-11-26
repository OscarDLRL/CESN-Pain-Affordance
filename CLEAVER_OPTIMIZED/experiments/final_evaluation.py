"""Classification-only comparison; matched observation schedules and latent-only timing."""
import argparse
import json
import time
import numpy as np
import torch
from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn import model as cesn
from CLEAVER_OPTIMIZED.detector import deployment_perception, fit_deployment_cesn
from CLEAVER_OPTIMIZED.experiments.discrete_inference import Incremental, fixed_losses, history, perceived_xy

FRAMES=[1,2,3,5,8,10,12,15,23,34,45,67,90,111,140,170,222]
STRIDES=[1,5,10]

def features(d,n,k,xy):
    y=d['yn'][:,:n:k]
    return torch.cat([d['static_n'],xy,y[:,-1],y.mean(1),torch.ones_like(y[:,:1,0])],1)

def ridge(A,label):
    A=A.double(); Y=torch.as_tensor(2*label-1,device=A.device,dtype=A.dtype)
    R=10*torch.eye(A.shape[1],device=A.device,dtype=A.dtype); R[-1,-1]=0
    return torch.linalg.solve(A.T@A+R,A.T@Y).float()

def nearest_features(d,n,k,xy):
    # All methods have access to the same visual and spatial inputs.
    y=d['yn'][:,:n:k].flatten(1)
    return torch.cat([d['static_n']/d['static_n'].shape[1]**.5,xy/2**.5,y/y.shape[1]**.5],1)

def timed(fn,reps=30,prepare=lambda:None):
    for _ in range(3): prepare(); fn()
    torch.cuda.synchronize(); samples=[]
    for _ in range(reps):
        prepare(); torch.cuda.synchronize()
        t=time.perf_counter(); fn(); torch.cuda.synchronize()
        samples.append((time.perf_counter()-t)*1000)
    return float(np.median(samples)),float(np.quantile(samples,.95))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--protocol',choices=['heldout_grid','offgrid'],default='heldout_grid'); a=ap.parse_args()
    device=torch.device('cuda'); torch.manual_seed(cfg.SEED); torch.set_float32_matmul_precision('high')
    off=a.protocol=='offgrid'; out=cfg.OUT/'final'/('interpolation/results' if off else 'results'); out.mkdir(parents=True,exist_ok=True)
    model,info,lo,hi,d=fit_deployment_cesn(device,protocol=a.protocol)
    tr,te=d['train'],d['test']; perception=deployment_perception(a.protocol)
    xt,_=perceived_xy(tr,perception,lo,hi,cfg.RGB)
    xv,p=perceived_xy(te,perception,lo,hi,cfg.OFFGRID if off else cfg.RGB)
    labels=te['label']; methods=[]; hist=[]; cost=[]
    spatial=np.linalg.norm(p-te['xy'],axis=1)*1000
    cesn.csv_write(out/'perception.csv',[dict(trajectory=key,error_mm=float(e)) for key,e in zip(te['keys'],spatial)])
    one={key:val[:1] for key,val in te.items()}; x1=xv[:1]
    def row(name,n,k,pred):
        pred=pred.cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
        return dict(method=name,frames=n,obs_every=k,observations=(n-1)//k+1,accuracy_pct=100*float(np.mean(pred==labels)),pain_recall_pct=100*float(np.mean(pred[labels==1]==1)),safe_accuracy_pct=100*float(np.mean(pred[labels==0]==0)))
    for k in STRIDES:
        losses=history(model,te,xv,k)
        for n in FRAMES:
            L=fixed_losses(model,te,xv,n,k); inc=losses[:,n-1]
            torch.testing.assert_close(L,inc,rtol=3e-4,atol=3e-5)
            # Tie-breaking is identical (argmin selects SAFE).
            batch=row('cesn_batch',n,k,L.argmin(1)); online=row('cesn_incremental',n,k,inc.argmin(1))
            methods.extend([batch,online]); hist.append({key:v for key,v in online.items() if key!='method'})
            R=nearest_features(tr,n,k,xt); W=ridge(features(tr,n,k,xt),tr['label'])
            j=torch.cdist(nearest_features(te,n,k,xv),R).argmin(1).cpu().numpy()
            methods.append(row('nearest',n,k,tr['label'][j])); methods.append(row('linear',n,k,(features(te,n,k,xv)@W>0).long()))
            # Time one actual observation at the end of an equally long prefix.
            # Incremental state is prepared outside the timer, as in deployment.
            if not off and n==111:
                state=Incremental(model,one,x1,k)
                for t in range(n-1): state.step(one['yn'][:,t] if t%k==0 else None)
                saved=(state.r.clone(),state.prev.clone(),state.sq.clone(),state.t,state.count)
                def restore():
                    state.r,state.prev,state.sq,state.t,state.count=saved[0],saved[1],saved[2].clone(),saved[3],saved[4]
                def incremental():
                    return state.step(one['yn'][:,n-1]).argmin(1)
                jobs={'cesn_batch':lambda:fixed_losses(model,one,x1,n,k).argmin(1), 'cesn_incremental':incremental,
                      'nearest':lambda:torch.cdist(nearest_features(one,n,k,x1),R).argmin(1), 'linear':lambda:(features(one,n,k,x1)@W>0)}
                for name,fn in jobs.items():
                    ms,p95=timed(fn,prepare=restore if name=='cesn_incremental' else lambda:None); cost.append(dict(method=name,frames=n,obs_every=k,ms=ms,p95_ms=p95,scope='latent_only'))
            print(f'k={k} n={n}: '+ ' | '.join(f"{r['method']} {r['accuracy_pct']:.1f}%" for r in methods[-4:]),flush=True)
    cesn.csv_write(out/'history.csv',hist); cesn.csv_write(out/'methods.csv',methods)
    if cost: cesn.csv_write(out/'cost.csv',cost)
    sim=cfg.OUT/('sim_closed_loop_offgrid' if off else 'sim_closed_loop')/'summary.json'
    if sim.exists():
        s=json.loads(sim.read_text()); cesn.atomic_json(out/'deployment_latency.json',dict(ms_per_step=s['ms_per_step'],ms_per_observation=s['ms_per_observation'],scope='detector including dynamic encoder on observation steps; excludes rendering and reset',source=str(sim)))
    print('DONE',out,flush=True)
if __name__=='__main__': main()
