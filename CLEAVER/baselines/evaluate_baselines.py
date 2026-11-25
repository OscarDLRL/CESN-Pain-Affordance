"""Frozen CLEAVER comparison. Run as CLEAVER.baselines.evaluate_baselines."""
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
import torch
from CLEAVER.config import EXPLICIT_DIR, OUT, TRAIN, TEST
from CLEAVER.core import train_explicit as tr
from CLEAVER.analysis import invert_explicit as inv
from CLEAVER.core.cesn import ContextualESNTorch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--grid-mm', type=float, default=2.)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--restarts', type=int, default=9)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-iter', type=int, default=80)
    p.add_argument('--frames', nargs='*', type=int, default=[1,2,3,5,10,12])
    p.add_argument('--percent', nargs='*', type=float, default=[5,10,15,20,25,30,40,50,60,70,80,90,100])
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    assert args.grid_mm > 0 and min(args.batch,args.restarts,args.max_iter)>0
    assert all(0<x<=100 for x in args.percent)
    assert all(x>0 for x in args.frames)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(EXPLICIT_DIR/'cesn.pt',map_location='cpu')
    assert set(ck['train_pairs'])==set(TRAIN) and set(ck['test_pairs'])==set(TEST)
    assert not set(TRAIN)&set(TEST)
    model = ContextualESNTorch.from_state_dict(ck['model_state'],device=device)
    train = tr.load_raw_sequences(set(TRAIN))
    test = inv.load_test_sequences(ck['test_pairs'])
    assert len(train)==140 and len(test)==22
    mean=ck['pca_mean'].cpu().numpy().astype(np.float32)
    comp=ck['pca_components'].cpu().numpy().astype(np.float32)
    def static(z): return inv.transform_static(z,mean,comp)
    def tensor(x): return torch.as_tensor(x,device=model.device,dtype=model.dtype)
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    train_xy=np.array([[r['grasp_x'],r['grasp_y']] for r in train])
    lo=train_xy.min(0); hi=train_xy.max(0)
    assert np.all(hi>lo)
    assert max(abs(lo[0]),abs(hi[0]))<inv.GRASP_X_SCALE and max(abs(lo[1]),abs(hi[1]))<inv.GRASP_Y_SCALE
    for r in test:
        xy=np.array([r['true_grasp_x'],r['true_grasp_y']]); assert np.all(xy>=lo-1e-6) and np.all(xy<=hi+1e-6)
    grid=np.array(np.meshgrid(*[np.linspace(a,b,math.ceil((b-a)*1000/args.grid_mm)+1) for a,b in zip(lo,hi)],indexing='xy')).reshape(2,-1).T
    starts=np.random.default_rng(args.seed).uniform(lo,hi,(args.restarts,2))
    T=len(train[0]['z_dynamic']); assert all(len(r['z_dynamic'])==T for r in train+test)
    ns=sorted(set(args.frames+[math.ceil(T*x/100) for x in args.percent])); assert all(n<=T for n in ns)
    if args.quick: test=test[:2]; ns=[1,10]
    train_z=tensor(np.stack([r['z_dynamic'] for r in train])); train_s=tensor(np.stack([static(r['z_static']) for r in train]))
    # Equal weighting of static and motion blocks; scale estimated on training only.
    zs=max(float(train_z.var(unbiased=False).item()),1e-12); ss=max(float(train_s.var(unbiased=False).item()),1e-12)
    low=tensor(lo); high=tensor(hi)
    def predict(label, s, xy, n):
        base=inv.build_base_context_batch(label,s,xy[:,0],xy[:,1],model)
        return inv.predict_batch(model,base,n)
    def grid_search(obs,s):
        best=[]
        with torch.no_grad():
            for label in (0,1):
                winner=None
                for i in range(0,len(grid),args.batch):
                    xy=tensor(grid[i:i+args.batch]); loss=((predict(label,s,xy,len(obs))-obs[None])**2).mean((1,2)); j=int(loss.argmin())
                    cand=(float(loss[j]),label,grid[i+j].copy())
                    if winner is None or cand[0]<winner[0]: winner=cand
                best.append(winner)
        return best,2*len(grid)
    def lbfgs(obs,s):
        best=[]; calls=0
        for label in (0,1):
            u=((tensor(starts)-low)/(high-low)*2-1).clamp(-.999999,.999999)
            raw=torch.atanh(u).detach().requires_grad_(True)
            opt=torch.optim.LBFGS([raw],lr=1.,max_iter=args.max_iter,max_eval=2*args.max_iter,history_size=20,tolerance_grad=1e-7,tolerance_change=1e-9,line_search_fn='strong_wolfe')
            def xy_now(): return low+(torch.tanh(raw)+1)*.5*(high-low)
            def closure():
                nonlocal calls
                calls+=args.restarts
                opt.zero_grad(set_to_none=True)
                loss=((predict(label,s,xy_now(),len(obs))-obs[None])**2).mean((1,2)).sum()
                loss.backward(); return loss
            opt.step(closure)
            with torch.no_grad():
                xy=xy_now(); loss=((predict(label,s,xy,len(obs))-obs[None])**2).mean((1,2)); j=int(loss.argmin()); calls+=args.restarts
                best.append((float(loss[j]),label,xy[j].cpu().numpy().copy()))
        return best,calls
    def nearest(obs,s):
        with torch.no_grad():
            d=((train_z[:,:len(obs)]-obs[None])**2).mean((1,2))/zs+((train_s-tensor(s))**2).mean(1)/ss
            best=[]
            for label in (0,1):
                ids=[i for i,r in enumerate(train) if int(r['pain_label'])==label]
                j=ids[int(d[ids].argmin())]; best.append((float(d[j]),label,train_xy[j].copy(),j))
        return best,len(train)
    out=OUT/'baselines'/('quick' if args.quick else f'grid{args.grid_mm:g}mm_seed{args.seed}')
    out.mkdir(parents=True,exist_ok=True)
    (out/'settings.json').write_text(json.dumps(dict(vars(args),bounds_m=[lo.tolist(),hi.tolist()],grid_candidates_per_label=len(grid),device=str(device),nearest_distance='dynamic prefix MSE / train variance + static PCA MSE / train variance',timing='synchronized online inference, excludes future rollout and shared input preparation'),indent=2))
    rows=[]
    for n in ns:
        for i,r in enumerate(test):
            s=static(r['z_static']); z=tensor(r['z_dynamic']); obs=z[:n]; label=int(r['pain_label']); xy_true=np.array([r['true_grasp_x'],r['true_grasp_y']]);
            for name,method in [('Nearest demonstration',nearest),('Dense grid search',grid_search),('CESN L-BFGS',lbfgs)]:
                sync(); t=time.perf_counter(); candidates,calls=method(obs,s); sync(); ms=(time.perf_counter()-t)*1000
                assert all(np.isfinite(c[0]) for c in candidates)
                winner=min(candidates,key=lambda c:c[0]); loss,lab,xy=winner[:3]; good=candidates[label][0]; bad=candidates[1-label][0]
                future=float('nan')
                if n<T:
                    with torch.no_grad():
                        pred=train_z[winner[3]] if name=='Nearest demonstration' else predict(lab,s,tensor(xy[None]),T)[0]
                        future=float(((pred[n:]-z[n:])**2).mean().sqrt())
                rows.append(dict(method=name,trajectory=i,pair_id=r.get('pair_id',''),frames=n,observed_percent=100*n/T,true_label=label,pred_label=lab,correct=int(lab==label),xy_error_mm=float(np.linalg.norm(xy-xy_true)*1000),future_rmse=future,inference_ms=ms,candidate_evaluations=calls,margin=(bad-good)/max(good+bad,1e-12),safe_loss=candidates[0][0],pain_loss=candidates[1][0],precontact=int(n<=r['pair_contact_frame']) if 'pair_contact_frame' in r else ''))
        for name in ('Nearest demonstration','Dense grid search','CESN L-BFGS'):
            rr=[x for x in rows if x['frames']==n and x['method']==name]; print(f'{n:3d} frames | {name:22s} | acc={np.mean([x["correct"] for x in rr])*100:6.2f}% | XY={np.mean([x["xy_error_mm"] for x in rr]):7.3f} mm',flush=True)
        with (out/'per_trajectory.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('Saved:',out)
if __name__=='__main__': main()
