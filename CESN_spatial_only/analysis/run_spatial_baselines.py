"""Evaluate fixed-grid and nearest-training-prefix baselines; no future frames."""
import argparse
import numpy as np
import pandas as pd
import torch
from CESN_spatial_only.analysis.invert_spatial_only import (
    load_test_sequences, transform_static, build_context_sequence, predict_batch,
    HISTORIES, xy_error_mm, ContextualESNTorch)
from CESN_spatial_only.core.parameters_spatial_only import (
    OUT_DIR, MODEL_PATH, GRASP_X_SCALE, GRASP_Y_SCALE)

@torch.no_grad()
def grid_search(model, static, obs, grid, batch):
    target = torch.as_tensor(obs, device=model.device, dtype=model.dtype)
    best = (float('inf'), None)
    for start in range(0,len(grid),batch):
        xy = torch.as_tensor(grid[start:start+batch], device=model.device, dtype=model.dtype)
        u = build_context_sequence(static, xy[:,0], xy[:,1], len(obs), model.device)
        loss = (predict_batch(model,u)-target[None]).square().mean((1,2))
        i = int(loss.argmin().item()); v = float(loss[i].item())
        if v < best[0]: best = (v, grid[start+i])
    return best[1]

def nearest(obs, train):
    # Compare equal-length prefixes at the same causal frame indices.
    # Shorter demonstrations are held at their last available frame.
    def score(d):
        z = d['z_dynamic']; idx = np.minimum(np.arange(len(obs)),len(z)-1)
        return float(np.mean((z[idx]-obs)**2))
    d = min(train,key=score)
    return (d['true_grasp_x'],d['true_grasp_y']), d['key']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--grid-mm',type=float,default=5); p.add_argument('--batch',type=int,default=128); a=p.parse_args()
    if a.grid_mm<=0 or a.batch<=0: p.error('positive grid spacing and batch required')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=torch.load(MODEL_PATH,map_location='cpu'); model=ContextualESNTorch.from_state_dict(ck['model_state'],device=device)
    assert not set(ck['train_pairs']) & set(ck['test_pairs'])
    train=load_test_sequences(ck['train_pairs']); test=load_test_sequences(ck['test_pairs'])
    assert len(train)==140 and len(test)==22
    m=ck['pca_mean'].numpy(); c=ck['pca_components'].numpy()
    step=a.grid_mm/1000
    xs=np.linspace(-GRASP_X_SCALE,GRASP_X_SCALE,int(np.ceil(2*GRASP_X_SCALE/step))+1)
    ys=np.linspace(-GRASP_Y_SCALE,GRASP_Y_SCALE,int(np.ceil(2*GRASP_Y_SCALE/step))+1)
    grid=np.array([(x,y) for x in xs for y in ys],dtype=np.float32)
    print(f'Grid: {len(grid)} candidates; spacing <= {a.grid_mm:g} mm',flush=True)
    rows=[]
    for item in test:
        static=transform_static(item['z_static'],m,c); z=item['z_dynamic']; T=len(z)
        for pct in HISTORIES:
            n=min(T,max(2,int(np.ceil(T*pct/100))))
            obs=z[:n]; xy=grid_search(model,static,obs,grid,a.batch); nn,key=nearest(obs,train)
            for name,coords,demo in [('Grid search',xy,''),('Nearest demonstration',nn,key)]:
                x,y=map(float,coords); label=int(x<item['knife_x'])
                rows.append(dict(method=name,trajectory=item['key'],history_percent=pct,n_obs=n,pred_x=x,pred_y=y,true_label=item['pain_label'],pred_label=label,correct=int(label==item['pain_label']),xy_error_mm=xy_error_mm(x,y,item['true_grasp_x'],item['true_grasp_y']),nearest_key=demo))
        print('DONE',item['key'],flush=True)
    dest=OUT_DIR/'baselines'; dest.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows); df.to_csv(dest/'baseline_details.csv',index=False)
    s=df.groupby(['method','history_percent']).agg(xy_mean_mm=('xy_error_mm','mean'),xy_std_mm=('xy_error_mm','std'),affordance_accuracy_pct=('correct',lambda x:100*x.mean())).reset_index()
    s.to_csv(dest/'baseline_summary.csv',index=False); print(s.to_string(index=False))
if __name__=='__main__': main()
