"""Rearrange the current package without modifying training or checkpoint files."""
import argparse
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(); p.add_argument('--apply',action='store_true'); a=p.parse_args()
root=Path(__file__).resolve().parent; pkg=root/'CLEAVER_OPTIMIZED'
moves={'dimension_cesn_core.py':'cesn/model.py','run_dimensions.py':'experiments/run_dimensions.py','run_forward_lowleak.py':'experiments/run_forward_lowleak.py'}
repl={'from CLEAVER_OPTIMIZED import dimension_cesn_core as b':'from CLEAVER_OPTIMIZED.cesn import model as b','from CLEAVER_OPTIMIZED.run_dimensions import':'from CLEAVER_OPTIMIZED.experiments.run_dimensions import'}
for src,dst in moves.items():
    if not (pkg/src).is_file() or (pkg/dst).exists(): raise SystemExit(f'Unexpected path: {src} -> {dst}; no changes made')
changes={}
for f in pkg.rglob('*.py'):
    if 'outputs' in f.parts: continue
    old=f.read_text(); new=old
    for x,y in repl.items(): new=new.replace(x,y)
    if new!=old:
        compile(new,str(f),'exec'); changes[f]=new
for src,dst in moves.items(): print('MOVE',src,'->',dst)
for f in changes: print('UPDATE IMPORTS',f.relative_to(root))
if not a.apply: print('Dry run only. Use --apply after committing and pushing existing code.'); raise SystemExit(0)
# Require the original sources to be tracked and unchanged before relocating them.
for src in moves:
    path=str((pkg/src).relative_to(root)); subprocess.run(['git','ls-files','--error-unmatch',path],cwd=root,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--exit-code','HEAD','--',path],cwd=root,check=True)
for src,dst in moves.items():
    target=pkg/dst; target.parent.mkdir(parents=True,exist_ok=True); (target.parent/'__init__.py').touch(exist_ok=True)
    subprocess.run(['git','mv',str((pkg/src).relative_to(root)),str(target.relative_to(root))],cwd=root,check=True)
for f,new in changes.items():
    dest=pkg/moves.get(str(f.relative_to(pkg)),str(f.relative_to(pkg))); dest.write_text(new)
print('Reorganized. Review git diff HEAD and verify imports before committing. Existing output manifests are historical; use a new run-name for future sweeps.')
