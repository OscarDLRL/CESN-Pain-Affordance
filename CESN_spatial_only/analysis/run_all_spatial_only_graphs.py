import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from CESN_spatial_only.core.parameters_spatial_only import OUT_DIR


def main():
    d=pd.read_csv(OUT_DIR/'inversion'/'spatial_only_per_trajectory.csv'); d['method']='L-BFGS'
    b=pd.read_csv(OUT_DIR/'baselines'/'baseline_details.csv')
    keys=['trajectory','history_percent','n_obs']
    for name,g in b.groupby('method'):
        assert not g.duplicated(keys).any() and set(map(tuple,g[keys].values))==set(map(tuple,d[keys].values)),name
    all_data=pd.concat([d,b],ignore_index=True); dest=OUT_DIR/'paper_figures'; dest.mkdir(parents=True,exist_ok=True)
    s=all_data.groupby(['method','history_percent']).agg(mean=('xy_error_mm','mean'),sd=('xy_error_mm','std'),accuracy=('correct',lambda x:100*x.mean()),p90=('xy_error_mm',lambda x:np.percentile(x,90)),within10=('xy_error_mm',lambda x:100*(x<=10).mean())).reset_index()
    s.to_csv(dest/'comparison_summary.csv',index=False)
    def save(name):
        plt.xlabel('Observed trajectory (%)'); plt.grid(alpha=.2); plt.legend(); plt.tight_layout()
        for ext in ['png','pdf']: plt.savefig(dest/f'{name}.{ext}',dpi=300)
        plt.close()
    for metric,ylabel,name in [('accuracy','Affordance accuracy (%)','01_affordance_accuracy'),('mean','Grasp XY error (mm)','02_xy_mean_sd'),('p90','90th percentile XY error (mm)','03_xy_p90'),('within10','Trajectories within 10 mm (%)','04_xy_success_10mm')]:
        plt.figure(figsize=(8,5))
        for method,g in s.groupby('method'):
            x=g.history_percent.to_numpy(); y=g[metric].to_numpy(); line,=plt.plot(x,y,marker='o',label=method)
            if metric=='mean':
                sd=g.sd.to_numpy(); plt.fill_between(x,np.maximum(0,y-sd),y+sd,alpha=.15,color=line.get_color())
        if metric in ['accuracy','within10']: plt.ylim(0,103)
        if metric=='mean': plt.axhline(10,ls='--',color='gray',label='10 mm'); plt.title('Mean ± 1 SD (lower band clipped at zero)')
        plt.ylabel(ylabel); save(name)
    print(s.to_string(index=False)); print('Saved figures:',dest)
if __name__=='__main__': main()
