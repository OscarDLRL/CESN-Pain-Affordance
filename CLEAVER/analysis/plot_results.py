from CLEAVER.config import EXPLICIT_DIR, SPATIAL_DIR, OUT
import pandas as pd
import matplotlib.pyplot as plt

a=pd.read_csv(EXPLICIT_DIR/'inversion/summary.csv')
b=pd.read_csv(SPATIAL_DIR/'inversion/spatial_only_summary.csv')
out=OUT/'figures'; out.mkdir(exist_ok=True)
fig,ax=plt.subplots(); ax.plot(a.history_percent,100*a.joint_task_accuracy,'o-',label='Explicit CESN'); ax.plot(a.history_percent,100*a.precontact_task_accuracy,'s--',label='Pre-contact subset'); ax.axhline(50,ls=':',label='Balanced constant baseline'); ax.set(xlabel='Observed trajectory (%)',ylabel='Intent accuracy (%)',ylim=(0,105)); ax.legend(); fig.tight_layout(); fig.savefig(out/'intent_accuracy.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(); ax.plot(a.history_percent,a.joint_xy_mean_mm,'o-',label='Explicit'); ax.plot(b.history_percent,b.xy_mean_mm,'s-',label='Spatial only'); ax.set(xlabel='Observed trajectory (%)',ylabel='Mean XY error (mm)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'xy_error.png',dpi=180); plt.close(fig)
