"""Actual LgDog physics backends. Neither backend instantiates a PPO runner."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .config import lgdog1_spec


def resolve_robot_xml(mjlab_root, robot_xml=None):
    if robot_xml:
        path = Path(robot_xml).expanduser().resolve()
    else:
        from .adapters.mjlab_tracking import add_mjlab_project
        project = add_mjlab_project(mjlab_root)
        path = project / "src/assets/robots/lgdog1/xmls/lgdog1.xml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_lgdog_model(path, visuals=False):
    """Keep the MJCF inertials/collisions; omit visual meshes for headless CPU use.

    PD values match the current LgDog actuator configuration: 20/1,20/1,40/2;
    hip/thigh torque 10 Nm, calf 20 Nm; calf armature .02, others .01.
    """
    import mujoco
    root = ET.parse(path).getroot()
    if visuals:
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.set("meshdir", str((Path(path).parent / compiler.get("meshdir", ".")).resolve()))
    else:
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "mesh" or (child.tag == "geom" and child.get("type") == "mesh"):
                    parent.remove(child)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", ".002")
    option.set("integrator", "implicitfast")
    world = root.find("worldbody")
    floor = ET.SubElement(world, "geom", name="supermimic_floor", type="plane", size="0 0 .1",
                          contype="1", conaffinity="1", friction=".6 .3 .3")
    if visuals:
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "texture", name="sm_checker", type="2d", builtin="checker",
                      rgb1=".22 .27 .34", rgb2=".35 .41 .49", width="256", height="256")
        ET.SubElement(asset, "material", name="sm_floor_material", texture="sm_checker", texrepeat="4 4")
        floor.set("material", "sm_floor_material")
        ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true",
                      diffuse=".8 .8 .8", ambient=".3 .3 .3")
    actuators = root.find("actuator")
    if actuators is not None:
        root.remove(actuators)
    actuators = ET.SubElement(root, "actuator")
    robot = lgdog1_spec()
    for name in robot.joint_names:
        calf = "calf" in name
        joint = root.find(f".//joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"robot XML is missing {name}")
        joint.set("armature", ".02" if calf else ".01")
        ET.SubElement(actuators, "position", name=f"sm_{name}", joint=name,
                      kp="40" if calf else "20", kv="2" if calf else "1",
                      forcerange="-20 20" if calf else "-10 10")
    for geom in root.findall(".//geom"):
        if "collision" in geom.get("name", ""):
            foot = "calf1_collision" in geom.get("name", "")
            geom.set("contype", "1")
            geom.set("conaffinity", "0")
            geom.set("condim", "3" if foot else "1")
            if foot:
                geom.set("priority", "1")
                geom.set("friction", ".6 .3 .3")
                geom.set("solimp", ".9 .95 .023")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class MujocoSimulator:
    def __init__(self, xml, num_envs, device, visuals=False):
        import mujoco
        self.model = load_lgdog_model(xml, visuals=visuals)
        if self.model.nq != 19 or self.model.nv != 18:
            raise ValueError("expected a free root and 12 hinge joints")
        self.data = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.device, self.num_envs = device, num_envs
        self.default = np.array(lgdog1_spec().default_joint_pos)
        self.qadr, self.vadr = joint_addresses(self.model)
        self.context = torch.zeros(num_envs, 16, device=device)
        self.context[:, 0] = .6

    def state(self):
        result = [np.concatenate((d.qpos[:7], d.qpos[self.qadr], d.qvel[:6], d.qvel[self.vadr])) for d in self.data]
        return torch.tensor(np.array(result), dtype=torch.float32, device=self.device)

    def reset(self, ids, states):
        import mujoco
        for i, s in zip(ids.tolist(), states.detach().cpu().numpy()):
            d = self.data[i]
            mujoco.mj_resetData(self.model, d)
            d.qpos[:7], d.qpos[self.qadr] = s[:7], s[7:19]
            d.qvel[:6], d.qvel[self.vadr] = s[19:25], s[25:37]
            d.ctrl[:] = s[7:19]
            mujoco.mj_forward(self.model, d)

    def step(self, actions):
        import mujoco
        targets = self.default + .25 * actions.detach().cpu().numpy()
        for d, target in zip(self.data, targets):
            d.ctrl[:] = target
            mujoco.mj_step(self.model, d, nstep=10)
            mujoco.mj_forward(self.model, d)
        return self.state()

    def close(self):
        pass


def joint_addresses(model):
    import mujoco
    ids = []
    for name in lgdog1_spec().joint_names:
        matches = [j for j in range(model.njnt) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").split("/")[-1] == name]
        if len(matches) != 1:
            raise ValueError(f"expected one joint {name}, got {matches}")
        ids.append(matches[0])
    return np.array(model.jnt_qposadr[ids]), np.array(model.jnt_dofadr[ids])


class MjlabSimulator:
    def __init__(self, mjlab_root, num_envs, device):
        if not str(device).startswith("cuda:"):
            raise ValueError("mjlab backend requires a CUDA device; use mujoco for CPU physics")
        from .adapters.mjlab_tracking import add_mjlab_project
        add_mjlab_project(mjlab_root)
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import MujocoCfg, SimulationCfg
        from mjlab.sim.sim import Simulation
        from mjlab.terrains import TerrainEntityCfg
        from src.assets.robots import get_lgdog1_robot_cfg
        self.device, self.num_envs = device, num_envs
        self.scene = Scene(SceneCfg(num_envs=num_envs, env_spacing=2.,
                                   entities={"robot": get_lgdog1_robot_cfg()},
                                   terrain=TerrainEntityCfg(terrain_type="plane")), device=device)
        self.sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=.002)),
                              model=self.scene.compile(), device=device)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        self.robot = self.scene["robot"]
        self.qadr, self.vadr = joint_addresses(self.sim.mj_model)
        self.qadr = torch.tensor(self.qadr, device=device)
        self.vadr = torch.tensor(self.vadr, device=device)
        self.jids = torch.tensor([self.robot.joint_names.index(n) for n in lgdog1_spec().joint_names], device=device)
        self.default = torch.tensor(lgdog1_spec().default_joint_pos, device=device)
        self.context = torch.zeros(num_envs, 16, device=device)
        self.context[:, 0] = .6  # Fixed nominal context: no DR events in this first online runner.
        self.sim.forward()

    def state(self):
        q, v = self.sim.data.qpos[:], self.sim.data.qvel[:]
        if q.shape[-1] != 19 or v.shape[-1] != 18:
            raise ValueError("expected exactly one LgDog robot per mjlab world")
        return torch.cat((q[:, :3] - self.scene.env_origins, q[:, 3:7], q[:, self.qadr],
                          v[:, :6], v[:, self.vadr]), dim=-1).clone()

    def reset(self, ids, states):
        self.sim.reset(ids)
        self.scene.reset(ids)
        q = self.sim.data.qpos[ids].clone()
        v = self.sim.data.qvel[ids].clone()
        q[:, :3] = states[:, :3] + self.scene.env_origins[ids]
        q[:, 3:7], q[:, self.qadr] = states[:, 3:7], states[:, 7:19]
        v[:, :6], v[:, self.vadr] = states[:, 19:25], states[:, 25:37]
        self.sim.data.qpos[ids], self.sim.data.qvel[ids] = q, v
        self.robot.set_joint_position_target(states[:, 7:19], joint_ids=self.jids, env_ids=ids[:, None])
        self.sim.forward()

    def step(self, actions):
        self.robot.set_joint_position_target(self.default + .25 * actions, joint_ids=self.jids)
        for _ in range(10):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(.002)
        self.sim.forward()
        return self.state()

    def close(self):
        # Scene/Simulation own no renderer; releasing the backend releases their tensors.
        pass
