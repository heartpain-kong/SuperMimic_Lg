"""How assets/g1 was made: export BlinkMimic's G1 tracking scene, i.e. the mjlab-compiled MJCF with its meshes, the
simulator options BlinkMimic applies at runtime, and the robot metadata (joint and body order, actuators, default pose,
action scale, domain-randomization targets, and the exact values of the fields the XML rounds).

Runs in BlinkMimic's environment (it needs mjlab), on the CPU:
    PYTHONPATH=<blinkmimic>/src python tools/export_g1_from_blinkmimic.py assets/g1
Writes scene.xml, assets/meshes/ and meta.json, and checks that the exported XML with the options applied compiles to the
same model as mjlab's (every compared array matches exactly, or to float32 rounding for the fields restored from meta.json).
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from mjlab.scene import Scene

from blinkmimic.env import make_scene_cfg, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.robot import get_robot

out = Path(sys.argv[1])
robot = get_robot("g1")
scene = Scene(make_scene_cfg(robot, 1), device="cpu")
sim_cfg = make_sim(robot)
ref = scene.compile()
sim_cfg.mujoco.apply(ref)
scene.write(out)
# this mjlab version writes only in-memory assets; the G1 meshes come from files next to the robot MJCF, so copy them by name
import shutil
import xml.etree.ElementTree as ET

src_dir = Path(robot.mjcf_path).parent
root = ET.parse(out / "scene.xml").getroot()
comp = root.find("compiler")
meshdir = out / (comp.get("meshdir", "") if comp is not None else "")
for elem in root.iter():
    f = elem.get("file")
    if f and not (meshdir / f).exists():
        hits = list(src_dir.rglob(Path(f).name))
        assert len(hits) == 1, (f, hits)
        (meshdir / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hits[0], meshdir / f)

model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
sim_cfg.mujoco.apply(model)

FIELDS = ["body_parentid", "body_jntnum", "body_jntadr", "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass",
          "body_inertia", "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_axis", "jnt_pos", "jnt_range", "jnt_limited",
          "dof_armature", "dof_damping", "dof_frictionloss", "geom_type", "geom_bodyid", "geom_contype", "geom_conaffinity",
          "geom_condim", "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_solref", "geom_solimp", "geom_margin",
          "geom_priority", "actuator_trnid", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
          "actuator_forcelimited", "actuator_ctrlrange", "actuator_ctrllimited", "mesh_vertnum", "qpos0"]
worst = {}
for f in FIELDS:
    a, b = np.asarray(getattr(ref, f), dtype=np.float64), np.asarray(getattr(model, f), dtype=np.float64)
    assert a.shape == b.shape, (f, a.shape, b.shape)
    worst[f] = float(np.abs(a - b).max()) if a.size else 0.0
for k in ("timestep", "integrator", "cone", "solver", "iterations", "tolerance", "ls_iterations", "ls_tolerance",
          "ccd_iterations", "impratio", "jacobian", "disableflags", "enableflags"):
    assert getattr(ref.opt, k) == getattr(model.opt, k), k
print("max |mjlab - exported| per field:", json.dumps(worst))
# the XML prints floats to ~7 significant digits (gains ~100 differ by ~5e-5): XMimic restores these fields from "exact"
EXACT = ["actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "dof_armature", "body_mass", "body_inertia", "body_ipos"]
bad = {f: v for f, v in worst.items()
       if v > 1e-6 * max(1.0, float(np.abs(np.asarray(getattr(ref, f), dtype=np.float64)).max(initial=0.0))) and f not in EXACT}
assert not bad, f"exported model differs: {bad}"

name = lambda obj, i: mujoco.mj_id2name(model, obj, i)  # noqa: E731
strip = lambda s: s.split("/", 1)[1] if s and "/" in s else s  # noqa: E731  (mjlab prefixes entity names with "robot/")
joint_names = list(robot.joint_names)
jid = {strip(name(mujoco.mjtObj.mjOBJ_JOINT, j)): j for j in range(model.njnt)}
act_of_joint = {}
for a in range(model.nu):
    act_of_joint[strip(name(mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a, 0])))] = a
bid = {strip(name(mujoco.mjtObj.mjOBJ_BODY, b)): b for b in range(model.nbody)}
free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
assert len(free) == 1
kin = Kinematics(robot, "cpu")
pelvis = int(model.jnt_bodyid[free[0]])
robot_geoms = [g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == pelvis]
ground_geoms = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
# BlinkMimic's DomainRandomizer selects the ground as the geoms of body 0; mjlab puts the plane on its own "terrain" body,
# so that selection is empty and the floor keeps its MJCF friction (contacts take the larger friction of the pair)
blink_ground = [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0]
meta = {
    "joint_names": joint_names,
    "qpos_adr": [int(model.jnt_qposadr[jid[n]]) for n in joint_names],
    "dof_adr": [int(model.jnt_dofadr[jid[n]]) for n in joint_names],
    "actuator_of_joint": [int(act_of_joint[n]) for n in joint_names],
    "free_qpos_adr": int(model.jnt_qposadr[free[0]]),
    "free_dof_adr": int(model.jnt_dofadr[free[0]]),
    "tracked_bodies": list(robot.tracked_bodies),
    "tracked_body_ids": [int(bid[n]) for n in robot.tracked_bodies],
    "com_body": robot.com_body,
    "com_body_id": int(bid[robot.com_body]),
    "com_range": [list(r) for r in robot.com_range],
    "robot_geom_ids": robot_geoms,
    "ground_geom_ids": ground_geoms,
    "blink_ground_geom_ids": blink_ground,
    "ground_friction": np.asarray(model.geom_friction[ground_geoms]).tolist(),
    "default_joint_pos": robot.default_joint_pos_list(),
    "action_scale": robot.action_scale_list(),
    "q_lo": kin.q_lo.tolist(),
    "q_hi": kin.q_hi.tolist(),
    "kin_body_names": kin.body_names,
    "kin_parent": kin.parent,
    "kin_joint_idx": kin.joint_idx,
    "kin_axes": kin.axes.tolist(),
    "kin_off_p": kin.off_p.tolist(),
    "kin_off_R": kin.off_R.tolist(),
    "kin_tracked_idx": kin.tracked_idx,
    "dr": {"friction_range": list(robot.dr.friction_range), "joint_offset": robot.dr.joint_offset,
           "reset_pos": list(robot.dr.reset_pos), "reset_rot": list(robot.dr.reset_rot),
           "reset_lin_vel": list(robot.dr.reset_lin_vel), "reset_ang_vel": list(robot.dr.reset_ang_vel),
           "reset_joint_pos": robot.dr.reset_joint_pos, "reset_joint_vel": robot.dr.reset_joint_vel},
    "sim": {"njmax": robot.njmax, "nconmax": robot.nconmax, "timestep": model.opt.timestep, "integrator": int(model.opt.integrator),
            "cone": int(model.opt.cone), "solver": int(model.opt.solver), "iterations": int(model.opt.iterations),
            "tolerance": float(model.opt.tolerance), "ls_iterations": int(model.opt.ls_iterations),
            "ls_tolerance": float(model.opt.ls_tolerance), "ccd_iterations": int(model.opt.ccd_iterations),
            "impratio": float(model.opt.impratio)},
    "counts": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "nbody": model.nbody, "ngeom": model.ngeom},
    "mujoco_version": mujoco.__version__,
    "exact": {f: np.asarray(getattr(ref, f), dtype=np.float64).tolist() for f in EXACT},
}
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print("geoms by type:", {int(t): int((model.geom_type[robot_geoms] == t).sum()) for t in np.unique(model.geom_type[robot_geoms])})
print("wrote", out, json.dumps(meta["counts"]), "options", json.dumps(meta["sim"]))
