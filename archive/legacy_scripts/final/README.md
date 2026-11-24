# Selected L-BFGS pipeline (planned)

Selected method: discrete Pick/Push hypotheses (Method A) with
batched L-BFGS (`max_iter=60`) on a frozen CESN.

No final experiment scripts are moved here by the phase-1 organizer.
Build a single validated `infer_context` entrypoint and then migrate
experiment scripts one by one, checking parity before retiring aliases.

Adam and Method B remain in `../comparisons` / `../legacy_adam` for
transparent comparison and reproducibility.
