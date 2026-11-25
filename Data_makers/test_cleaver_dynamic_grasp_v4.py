#!/usr/bin/env python3
"""Dynamic cleaver grasp/lift pilot. Requires test_knife_paired_reaches.py.
The virtual fixture resets the dynamic actor between control steps until release.
No object pose/velocity writes are allowed after release. Contact samples are
not a physical injury model. Only validated pairs are candidates for a dataset.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import sapien
from PIL import Image, ImageDraw
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from Data_makers import test_knife_paired_reaches as base

ENV_ID = 'DynamicCleaverGraspPilot-v4'
LENGTH, HEIGHT, THICKNESS = .18, .09, .003
BACK_Z, EDGE_BAND = .10, .004
EDGE_Z = BACK_Z + HEIGHT
HANDLE_LENGTH = .12
X, Y = .08, .05
ORIGIN = np.array([X,Y,BACK_Z+HEIGHT/2])
# Near the free end, so a horizontal palm need not pass through 9 cm of blade.
# Rotate the entire rigid body +90 degrees about world Z.
OBJECT_QUAT = np.array([np.sqrt(.5),0.,0.,np.sqrt(.5)])
GOAL = ORIGIN + np.array([0.,-LENGTH/2+.025,0.])
# Preserve the successful v4 common preparation in world coordinates.
COMMON_START = np.array([-.10,.05,.325])

@register_env(ENV_ID,max_episode_steps=1500)
class CleaverEnv(base.PainAffordanceKnifePairEnv):
    def _load_scene(self,options):
        self.table_scene=TableSceneBuilder(env=self,robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        b=self.scene.create_actor_builder()
        material=sapien.physx.PhysxMaterial(static_friction=1.0,dynamic_friction=.8,restitution=0.)
        def box(center,half,color):
            p=sapien.Pose(p=center)
            b.add_box_collision(pose=p,half_size=half,material=material)
            b.add_box_visual(pose=p,half_size=half,material=sapien.render.RenderMaterial(base_color=color))
        box([0,0,-EDGE_BAND/2],[LENGTH/2,THICKNESS/2,(HEIGHT-EDGE_BAND)/2],[.68,.70,.73,1])
        box([0,0,HEIGHT/2-EDGE_BAND/2],[LENGTH/2,THICKNESS/2,EDGE_BAND/2],[.85,.86,.88,1])
        box([LENGTH/2+HANDLE_LENGTH/2,0,-HEIGHT/2+.018],[HANDLE_LENGTH/2,.012,.014],[.10,.10,.11,1])
        b.initial_pose=sapien.Pose(p=ORIGIN,q=OBJECT_QUAT)
        self.obj=b.build(name='dynamic_cleaver')
    def _after_reconfigure(self,options):
        pass
    def _initialize_episode(self,env_idx,options):
        self.table_scene.initialize(env_idx)
        self.reset_fixture()
    def reset_fixture(self):
        self.obj.set_pose(sapien.Pose(p=ORIGIN,q=OBJECT_QUAT))
        self.obj.set_linear_velocity(np.zeros(3,dtype=np.float32))
        self.obj.set_angular_velocity(np.zeros(3,dtype=np.float32))


def entity_name(body):
    entity=getattr(body,'entity',body)
    return str(getattr(entity,'name',''))


def finger_side(name):
    """Group physical finger and fingertip links by jaw, not by link count."""
    name = name.lower()
    if 'finger' not in name:
        return None
    if 'left' in name:
        return 0
    if 'right' in name:
        return 1
    raise RuntimeError(f'Unrecognized finger side: {name!r}; inspect Panda link names')


def finger_links(e):
    groups = [[], []]
    for link in e.agent.robot.get_links():
        side = finger_side(link.name)
        if side is not None:
            groups[side].append(link)
    if not all(groups):
        raise RuntimeError(f'Missing finger jaw: {[l.name for l in e.agent.robot.get_links()]}')
    return groups


def contact_samples(e):
    # CPU PhysX contact points from the last physics step. Fail explicitly if
    # this installed SAPIEN version does not expose the API.
    scenes=getattr(e.scene,'sub_scenes',None)
    if scenes is None or not hasattr(scenes[0],'get_contacts'):
        raise RuntimeError('CPU contact API unavailable: inspect e.scene.sub_scenes[0].get_contacts')
    obj_name=entity_name(e.obj._objs[0])
    links=e.agent.robot.get_links()
    names={entity_name(x._objs[0]):x.name for x in links}
    pose=e.obj.pose.sp
    inv=pose.inv()
    edge_finger=edge_other=body_finger=other=0.
    for c in scenes[0].get_contacts():
        bodies=getattr(c,'bodies',None)
        if bodies is None:
            raise RuntimeError('Unsupported SAPIEN Contact: expected bodies')
        a,b=map(entity_name,bodies)
        if a==obj_name: link=names.get(b)
        elif b==obj_name: link=names.get(a)
        else: continue
        if link is None: continue
        finger=finger_side(link) is not None
        for pt in c.points:
            local=(inv*sapien.Pose(p=pt.position)).p
            impulse=float(np.linalg.norm(pt.impulse))
            # Local upper edge band; restrict to blade extent, not handle.
            edge=abs(local[0])<=LENGTH/2+.002 and local[2]>=HEIGHT/2-EDGE_BAND-.001
            if edge and finger: edge_finger+=impulse
            elif edge: edge_other+=impulse
            elif finger: body_finger+=impulse
            else: other+=impulse
    # Sum contact-force magnitudes over the links of each physical jaw.
    # This is a per-jaw contact indicator, not a net force vector.
    forces=[sum(float(np.linalg.norm(base.to_numpy(e.scene.get_pairwise_contact_forces(l,e.obj)))) for l in group) for group in finger_links(e)]
    return edge_finger,edge_other,body_finger,other,forces


class Capture(base.CaptureEnv):
    def __init__(self,env):
        self.fixture=True
        super().__init__(env)
    def clear(self):
        super().clear()
        self.contacts=[]; self.finger_forces=[]; self.object_quat=[]; self.fixture_flags=[]
    def reset(self,**kwargs):
        self.fixture=True
        return super().reset(**kwargs)
    def step(self,action):
        if self.fixture: self.unwrapped.reset_fixture()
        return super().step(action)
    def _append_observation(self,obs):
        super()._append_observation(obs)
        e=self.unwrapped
        *impulses,forces=contact_samples(e)
        self.contacts.append(impulses); self.finger_forces.append(forces)
        self.object_quat.append(np.asarray(e.obj.pose.sp.q).copy())
        self.fixture_flags.append(self.fixture)
    def episode(self):
        ep=super().episode()
        ep.update(contact_impulses=np.asarray(self.contacts),finger_forces=np.asarray(self.finger_forces),
                  object_quat=np.asarray(self.object_quat),fixture_active=np.asarray(self.fixture_flags))
        return ep

def slerp(a, b, t):
    a = np.array(a, dtype=float, copy=True); a /= np.linalg.norm(a)
    b = np.array(b, dtype=float, copy=True); b /= np.linalg.norm(b)
    dot = float(a @ b)
    if dot < 0: b, dot = -b, -dot
    dot = np.clip(dot, -1., 1.)
    if dot > .9995:
        q = (1-t)*a+t*b
        return q/np.linalg.norm(q)
    angle = np.arccos(dot)
    return (np.sin((1-t)*angle)*a+np.sin(t*angle)*b)/np.sin(angle)


def cartesian_plan(planner, env, points, quats, steps, knots=100):
    """Reproduce the feasibility test's linear Cartesian samples and SLERP.
    Time-resample each segment only after successful dense IK.
    """
    backend = planner.planner
    q = backend.pad_qpos(base.get_qpos(env).astype(float))
    idx = np.asarray(backend.move_group_joint_indices, dtype=int)
    mask = np.ones(len(q), dtype=bool); mask[idx] = False
    segments = []
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    n1 = int(np.clip(round(steps*lengths[0]/lengths.sum()), 2, steps-2))
    for seg in range(2):
        path = [q[idx].copy()]
        for k in range(1,knots+1):
            t = k/knots
            p = (1-t)*points[seg]+t*points[seg+1]
            quat = slerp(quats[seg],quats[seg+1],t)
            pose = planner._transform_pose_for_planning(sapien.Pose(p=p,q=quat))
            target = backend.transform_goal_to_wrt_base(np.r_[pose.p,pose.q])
            candidates = []
            for attempts in (1,20):
                status, solutions = backend.IK(target,q.copy(),mask=mask.tolist(),
                                                n_init_qpos=attempts,threshold=.001)
                if status == 'Success':
                    for v in solutions:
                        v = np.asarray(v,float)
                        if v.shape == q.shape and np.isfinite(v).all():
                            d = v[idx]-q[idx]
                            candidates.append((float(np.max(np.abs(d))),float(np.linalg.norm(d)),v.copy()))
                if candidates and min(c[0] for c in candidates)<=.15:
                    break
            if not candidates:
                raise RuntimeError(f'IK failed segment {seg+1}, knot {k}/{knots}: {status}; target={p.tolist()}')
            jump, _, qnew = min(candidates,key=lambda c:(c[0],c[1]))
            if jump>.15:
                raise RuntimeError(f'IK discontinuity segment {seg+1}, knot {k}/{knots}: {jump:.4f} rad')
            q=qnew.copy(); path.append(q[idx].copy())
        dense=np.asarray(path)
        n=(n1,steps-n1)[seg]
        t=np.linspace(0,1,n+1)
        # Smooth start/end of each straight segment, without extra hold frames.
        t=t*t*(3-2*t)
        sample=np.column_stack([np.interp(t,np.linspace(0,1,len(dense)),dense[:,j]) for j in range(len(idx))])
        segments.append(sample if seg==0 else sample[1:])
        print(f'IK segment {seg+1}: {knots}/{knots} complete',flush=True)
    commands=np.concatenate(segments,axis=0)
    jump=float(np.max(np.abs(np.diff(commands,axis=0))))
    print(f'Execution samples: {len(commands)}, max joint jump={jump:.4f} rad',flush=True)
    if jump>.15:
        raise RuntimeError('Execution jump exceeds .15 rad; increase --steps for both candidates')
    return commands


def resample(path,n):
    arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    keep = np.r_[True,np.diff(arc)>1e-10]
    arc,path = arc[keep],path[keep]
    if len(arc)<2: raise RuntimeError('Empty path')
    t=np.linspace(0,1,n); t=3*t*t-2*t*t*t
    return np.column_stack([np.interp(t*arc[-1],arc,path[:,j]) for j in range(7)])



def run(env,args,label,prep=None, start_candidate=None, plan_only=False, saved_plan=None):
    env.reset(seed=123)
    e=env.unwrapped
    vertical=e.agent.build_grasp_pose(np.array([0.,0.,-1.]),np.array([1.,0.,0.]),GOAL)
    qv=np.asarray(vertical.q,dtype=float)
    horizontal=e.agent.build_grasp_pose(np.array([0.,1.,0.]),np.array([1.,0.,0.]),GOAL)
    qh=np.asarray(horizontal.q,dtype=float)
    qstart=slerp(qv,qh,.5)
    planner=base.new_planner(env,vis=False)
    try:
        if prep is None:
            start=COMMON_START.copy() if start_candidate is None else np.asarray(start_candidate,float).copy()
            if not base.planner_step_ok(planner.move_to_pose_with_RRTConnect(sapien.Pose(p=start,q=qstart))):
                raise RuntimeError('Common preparation failed')
            planner.open_gripper()
            prep=np.array(env.actions,copy=True)
        else:
            for a in prep: env.step(a)
        open_cmd=float(prep[-1,-1])
        e.reset_fixture()
        env.restart_capture_from_current_state()
        start=base.get_tcp_xyz(env).astype(float)
        initial=np.asarray(e.agent.tcp.pose.sp.q,dtype=float)
        if label=='safe':
            waypoint=GOAL+np.array([0.,-args.standoff,0.])
            options=[qh]
        else:
            waypoint=np.array([GOAL[0],GOAL[1],EDGE_Z+.025])
            options=[qv]
        points=np.stack([start,waypoint,GOAL])
        initial_state=base.get_qpos(env).copy()
        if saved_plan is None:
            quat=options[0]
            commands=cartesian_plan(planner,env,points,[initial,quat,quat],args.steps,args.knots)
            cost=float(np.linalg.norm(np.diff(commands,axis=0),axis=1).sum())
            plan=dict(commands=commands,quat=quat,points=points,cost=cost,initial_qpos=initial_state)
            if plan_only:
                return plan,prep
        else:
            plan=saved_plan
            if np.max(np.abs(initial_state-plan['initial_qpos']))>1e-5:
                raise RuntimeError('Prepared state differs from the selected planning state')
            commands,quat,points=plan['commands'],plan['quat'],plan['points']
        for q in commands[1:]: env.step(np.r_[q,open_cmd].astype(np.float32))
        for _ in range(args.settle): env.step(np.r_[commands[-1],open_cmd].astype(np.float32))
        reach_end=len(env.rgb)-1
        # Close with the same commands for both trials, while fixture is active.
        for _ in range(args.close_steps): env.step(np.r_[commands[-1],-1.].astype(np.float32))
        close_end=len(env.rgb)-1
        # Plan vertical lift from the actual reached TCP, not the nominal target.
        pos=base.get_tcp_xyz(env).astype(float)
        actual_quat=np.asarray(e.agent.tcp.pose.sp.q,dtype=float)
        lift_points=np.stack([pos,pos+[0,0,args.lift/2],pos+[0,0,args.lift]])
        lift=cartesian_plan(planner,env,lift_points,[actual_quat]*3,args.lift_steps,args.knots)
        # No resets, constraints, or object pose writes after this point.
        env.fixture=False
        for q in lift[1:]: env.step(np.r_[q,-1.].astype(np.float32))
        for _ in range(args.hold_steps): env.step(np.r_[lift[-1],-1.].astype(np.float32))
        ep=env.episode()
        ep.update(reach_end=reach_end,close_end=close_end,goal=GOAL.copy(),waypoints=points,desired_grasp_quat=quat, planned_joint_length_rad=plan['cost'])
        return ep,prep
    finally: planner.close()


def summarize(ep,args):
    end=int(ep['reach_end']); close=int(ep['close_end']); c=ep['contact_impulses']; obj=ep['object_xyz']; tcp=ep['tcp_xyz']
    edge=c[:,0]+c[:,1]; hits=np.flatnonzero(edge>args.impulse_threshold)
    lift=float(obj[-1,2]-obj[close,2]); hold=obj[-args.hold_steps:] if args.hold_steps else obj[-1:]
    # Relative translation stability is a simple slip proxy, not grasp proof alone.
    rel=obj-tcp
    slip=float(np.max(np.linalg.norm(rel[close:]-rel[close],axis=1)))
    bilateral=bool(np.all(ep['finger_forces'][close]>args.force_threshold))
    return dict(frames=len(obj),reach_end=end,close_end=close,goal_error_m=float(np.linalg.norm(tcp[end]-GOAL)),
        first_edge_contact_frame=int(hits[0]) if len(hits) else None,edge_peak_impulse_Ns=float(edge.max()),
        edge_finger_peak_impulse_Ns=float(c[:,0].max()),edge_other_peak_impulse_Ns=float(c[:,1].max()),
        nonfinger_nonedge_peak_impulse_Ns=float(c[:,3].max()),bilateral_at_close=bilateral,
        object_lift_m=lift,minimum_hold_lift_m=float(hold[:,2].min()-obj[close,2]),relative_slip_m=slip,
        lift_success=bool(bilateral and lift>=.8*args.lift and hold[:,2].min()-obj[close,2]>=.8*args.lift and slip<=.02))


def select_pair(env,args):
    # A small local search, not a global optimum. All starts use the same
    # halfway quaternion and the same start for the two intentions.
    starts=[COMMON_START, COMMON_START+np.array([.05,-.05,0.]),
            COMMON_START+np.array([.05,-.10,.025])]
    records=[]; feasible=[]
    for i,start in enumerate(starts):
        row={'candidate':i,'requested_start_xyz_m':start.tolist()}
        try:
            a,prep=run(env,args,'safe',start_candidate=start,plan_only=True)
            b,_=run(env,args,'hazard',prep=prep,plan_only=True)
            if np.max(np.abs(a['initial_qpos']-b['initial_qpos']))>1e-5:
                raise RuntimeError('Pair initial joint states differ')
            ls,lp=a['cost'],b['cost']
            score=ls+lp+args.balance_weight*abs(ls-lp)
            row.update(feasible=True,safe_length_rad=ls,pain_length_rad=lp,score=score)
            feasible.append((score,i,prep,a,b))
        except RuntimeError as exc:
            row.update(feasible=False,reason=str(exc))
        records.append(row)
        print(json.dumps(row),flush=True)
    report={'candidates':records,'balance_weight':args.balance_weight,'selected':None}
    if feasible:
        best=min(feasible,key=lambda v:v[0]); report['selected']=best[1]
    (args.out/'start_search.json').write_text(json.dumps(report,indent=2)+'\n')
    if not feasible: raise RuntimeError('No shared 45-degree start passed both approach plans; see start_search.json')
    _,i,prep,a,b=best
    print(f'Selected shared start {i}; executing physical grasp/lift',flush=True)
    return prep,a,b


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preview',action='store_true')
    p.add_argument('--balance-weight',type=float,default=.5)
    p.add_argument('--steps',type=int,default=100); p.add_argument('--knots',type=int,default=100)
    p.add_argument('--settle',type=int,default=12); p.add_argument('--close-steps',type=int,default=30)
    p.add_argument('--lift-steps',type=int,default=60); p.add_argument('--hold-steps',type=int,default=20)
    p.add_argument('--lift',type=float,default=.05); p.add_argument('--standoff',type=float,default=.07)
    p.add_argument('--force-threshold',type=float,default=.1); p.add_argument('--impulse-threshold',type=float,default=.0001)
    p.add_argument('--out',type=Path,default=Path('datasets/cleaver_dynamic_grasp_v4_pilot'))
    args=p.parse_args()
    if min(args.steps,args.lift_steps)<4 or min(args.knots,args.close_steps,args.hold_steps)<1 or args.lift<=0: p.error('Invalid settings')
    args.out.mkdir(parents=True,exist_ok=False)
    raw=gym.make(ENV_ID,obs_mode='rgb',control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='physx_cpu',num_envs=1,max_episode_steps=1500)
    env=Capture(raw)
    try:
        obs,_=raw.reset(seed=123); Image.fromarray(base.observation_rgb(obs)).save(args.out/'scene256.png')
        if args.preview:return
        print('Finger links by jaw:', [[l.name for l in group] for group in finger_links(raw.unwrapped)], flush=True)
        prep,plan_a,plan_b=select_pair(env,args)
        a,_=run(env,args,'safe',prep=prep,saved_plan=plan_a); np.savez_compressed(args.out/'safe.npz',**a); base.save_gif(a['rgb'],args.out/'safe.gif')
        b,_=run(env,args,'hazard',prep=prep,saved_plan=plan_b); np.savez_compressed(args.out/'hazard.npz',**b); base.save_gif(b['rgb'],args.out/'hazard.gif')
        frames=[]
        for i,(left,right) in enumerate(zip(a['rgb'],b['rgb'])):
            im=Image.new('RGB',(512,280),'white'); im.paste(Image.fromarray(left),(0,24)); im.paste(Image.fromarray(right),(256,24))
            d=ImageDraw.Draw(im)
            for x,name,ep in [(4,'SAFE candidate',a),(260,'PAIN candidate',b)]:
                d.text((x,4),f'{name} | {ep["contact_impulses"][i,:2].sum():.4f} Ns',fill='black')
            frames.append(im)
        frames[0].save(args.out/'pair.gif',save_all=True,append_images=frames[1:],duration=67,loop=0)
        s=base.initial_pair_check(a,b); s.update(safe=summarize(a,args),hazard=summarize(b,args))
        s['planned_joint_length_rad']={'safe':float(a['planned_joint_length_rad']),'hazard':float(b['planned_joint_length_rad'])}
        s['actual_reach_joint_length_rad']={k:float(np.linalg.norm(np.diff(ep['qpos'][:int(ep['reach_end'])+1,:7],axis=0),axis=1).sum()) for k,ep in [('safe',a),('hazard',b)]}
        s['object_initial_quat_wxyz']=OBJECT_QUAT.tolist(); s['goal_xyz_m']=GOAL.tolist(); s['same_frame_count']=len(a['rgb'])==len(b['rgb'])
        s['final_reach_pair_error_m']=float(np.linalg.norm(a['tcp_xyz'][int(a['reach_end'])]-b['tcp_xyz'][int(b['reach_end'])]))
        s['accepted']=bool(s['same_frame_count'] and s['initial_rgb_mae']<=.25 and s['initial_qpos_max_abs']<=1e-5 and s['initial_object_xyz_max_abs']<=1e-6 and s['final_reach_pair_error_m']<=.005 and all(s[k]['goal_error_m']<=.005 and s[k]['lift_success'] for k in ('safe','hazard')) and s['safe']['first_edge_contact_frame'] is None and s['hazard']['first_edge_contact_frame'] is not None and s['hazard']['first_edge_contact_frame']<=s['hazard']['close_end'])
        s['note']='Temporary virtual fixture until close; unassisted dynamics during lift. Edge impulses sampled at control boundaries, not all physics substeps. No physical injury model.'
        (args.out/'summary.json').write_text(json.dumps(s,indent=2)+'\n'); print(json.dumps(s,indent=2))
    finally:env.close()

if __name__=='__main__':main()
