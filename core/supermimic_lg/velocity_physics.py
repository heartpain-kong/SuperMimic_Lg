"""Shared 50 Hz velocity-task physics for MBRL and policy comparisons."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from .config import lgdog1_spec
from .simulator import load_lgdog_model, joint_addresses
from .state import heading_quat

GRID = 9
SPACING = .25
TERRAINS = ("flat", "rough", "stairs_up", "stairs_down", "slope_up", "slope_down", "rough_height", "wave")
FAMILIES = dict(stairs_up="pyramid_stairs_inv", stairs_down="pyramid_stairs",
                slope_up="hf_pyramid_slope_inv", slope_down="hf_pyramid_slope",
                rough_height="random_rough", wave="wave_terrain")


def offsets(device="cpu"):
    a = torch.linspace(-1., 1., GRID, device=device)
    return torch.stack(torch.meshgrid(a, a, indexing="ij"), -1).flatten(0, 1)


def scene_config(root, terrain, n, seed, device, sensors):
    from .adapters.mjlab_tracking import add_mjlab_project
    add_mjlab_project(root)
    from mjlab.scene import SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from src.assets.robots import get_lgdog1_robot_cfg
    terrain_cfg = TerrainEntityCfg(terrain_type="plane")
    if terrain != "flat":
        from src.tasks.velocity.config.lgdog1.terrain_cfg import lgdog1_rough_terrain_cfg
        generator = lgdog1_rough_terrain_cfg()
        generator.seed = seed
        generator.curriculum = True
        if terrain != "rough":
            generator.sub_terrains = {FAMILIES[terrain]: generator.sub_terrains[FAMILIES[terrain]]}
            next(iter(generator.sub_terrains.values())).proportion = 1.
            generator.num_cols = 1
        terrain_cfg = TerrainEntityCfg(terrain_type="generator", terrain_generator=generator,
                                      max_init_terrain_level=1)
    return SceneCfg(num_envs=n, env_spacing=2., entities={"robot": get_lgdog1_robot_cfg()},
                    terrain=terrain_cfg, sensors=sensors)


def velocity_sensors():
    from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg, ContactSensorCfg, ContactMatch

    @dataclass
    class HighGrid(GridPatternCfg):
        def generate_rays(self, mj_model, device):
            # Native GridPatternCfg flattens XY; our stored patches
            # use row-X/column-Y. Explicitly share ordering with CPU.
            xy = offsets(device)
            p = torch.cat((xy, torch.full_like(xy[:, :1], 2.)), -1)
            d = torch.zeros_like(p); d[:, 2] = -1.
            return p, d

    scan = RayCastSensorCfg(name="velocity_ground", frame=ObjRef(type="body", name="trunk", entity="robot"),
            pattern=HighGrid(size=(2., 2.), resolution=SPACING), ray_alignment="yaw",
            include_geom_groups=(0,), max_distance=10., debug_vis=False)
    contacts = tuple(ContactSensorCfg(name=f"velocity_contact_{i}",
        primary=ContactMatch(mode="geom", pattern=name, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
        reduce="netforce", num_slots=1)
        for i, name in enumerate([f"{leg}_calf1_collision" for leg in ("FR", "FL", "RR", "RL")] + ["trunk_collision"]))
    return (scan, *contacts)


class VelocitySimulator:
    """Same geometry/PD/control timing for both algorithms; CPU and Warp backends.

    State = physical 37 + four foot contacts + trunk contact. Terrain observations
    are privileged world-model inputs; the deployed actor receives only HIM's
    47-feature proprioception/command frame.
    """
    def __init__(self, xml, root, n, backend, device, terrain="flat", seed=42, level=None, visuals=False):
        import mujoco
        if terrain not in TERRAINS or backend not in ("mujoco", "mjlab"):
            raise ValueError("unsupported velocity physics configuration")
        if backend == "mjlab" and not str(device).startswith("cuda:"):
            raise ValueError("mjlab requires CUDA")
        self.n, self.backend, self.device, self.terrain = n, backend, device, terrain
        self.scene = None
        self.rng = np.random.default_rng(seed)
        if backend == "mujoco" and terrain == "flat":
            self.model = load_lgdog_model(xml, visuals=visuals)
            self.origins = torch.zeros(n, 3, device=device)
        else:
            from mjlab.scene import Scene
            from .simulator import resolve_robot_xml
            if resolve_robot_xml(root).read_bytes() != Path(xml).read_bytes():
                raise ValueError("native terrain scene XML differs from the requested robot XML")
            sensors = ()
            if backend == "mjlab":
                sensors = velocity_sensors()
            self.scene = Scene(scene_config(root, terrain, n, seed, device, sensors), device=device)
            self.origins = self.scene.env_origins
            if level is not None and terrain != "flat":
                t = self.scene["terrain"]
                if not 0 <= level < t.terrain_origins.shape[0]:
                    raise ValueError("terrain level outside configured rows")
                t.terrain_levels[:] = level
                t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
            self.model = self.scene.compile()
        self.model.opt.timestep = .005  # Same as the existing LgDog HIM task.
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.ccd_iterations = 500
        self.qadr, self.vadr = joint_addresses(self.model)
        self.ctrl = []
        for name in lgdog1_spec().joint_names:
            matches = [i for i, j in enumerate(self.model.actuator_trnid[:, 0])
                       if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) or "").split("/")[-1] == name]
            if len(matches) != 1:
                raise ValueError(f"expected one PD actuator per joint: {name}")
            self.ctrl.append(matches[0])
        if (not np.allclose(self.model.actuator_gainprm[self.ctrl, 0], [20,20,40]*4)
                or not np.allclose(-self.model.actuator_biasprm[self.ctrl, 2], [1,1,2]*4)):
            raise ValueError("velocity task expects canonical LgDog PD gains")
        self.default = torch.tensor(lgdog1_spec().default_joint_pos, device=device)
        self.geom_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.contact_geoms = []
        for name in [f"{leg}_calf1_collision" for leg in ("FR", "FL", "RR", "RL")] + ["trunk_collision"]:
            self.contact_geoms.append(next(i for i in range(self.model.ngeom)
                    if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").split("/")[-1] == name))
        if backend == "mjlab":
            from mjlab.sim import MujocoCfg, SimulationCfg
            from mjlab.sim.sim import Simulation
            self.sim = Simulation(num_envs=n, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=.005, ccd_iterations=500),
                                 nconmax=35 if terrain != "flat" else None, contact_sensor_maxmatch=500),
                                 model=self.model, device=device)
            self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
            if self.scene.sensor_context is not None:
                self.sim.set_sensor_context(self.scene.sensor_context)
            self.robot = self.scene["robot"]
            self.jids = torch.tensor([self.robot.joint_names.index(name) for name in lgdog1_spec().joint_names], device=device)
            self.sim.forward()
        else:
            self.data = [mujoco.MjData(self.model) for _ in range(n)]
        self.reset(torch.arange(n, device=device))

    def reset(self, ids):
        import mujoco
        if not len(ids):
            return
        q = torch.zeros(len(ids), 19, device=self.device)
        q[:, :3] = self.origins[ids]
        q[:, 2] += .31
        q[:, 3] = 1.
        q[:, 7:] = self.default + torch.tensor(self.rng.uniform(-.003, .003, (len(ids), 12)), device=self.device)
        if self.backend == "mjlab":
            self.sim.reset(ids)
            self.scene.reset(ids)
            self.sim.data.qpos[ids] = q
            self.sim.data.qvel[ids] = 0.
            self.robot.set_joint_position_target(q[:, self.qadr], joint_ids=self.jids, env_ids=ids[:, None])
            self.sim.forward()
            self.scene.update(0.)
            self.sim.sense()
        else:
            for i, row in zip(ids.tolist(), q.cpu().numpy()):
                d = self.data[i]
                mujoco.mj_resetData(self.model, d)
                d.qpos[:7], d.qpos[self.qadr] = row[:7], row[7:]
                d.ctrl[self.ctrl] = row[7:]
                mujoco.mj_forward(self.model, d)

    def step(self, action):
        import mujoco
        targets = self.default + .25 * action.detach().clamp(-1., 1.)
        if self.backend == "mjlab":
            self.robot.set_joint_position_target(targets, joint_ids=self.jids)
            for _ in range(4):
                self.scene.write_data_to_sim()
                self.sim.step()
                self.scene.update(.005)
            self.sim.forward()
            self.scene.update(0.)
            self.sim.sense()
        else:
            for d, target in zip(self.data, targets.cpu().numpy()):
                d.ctrl[self.ctrl] = target
                mujoco.mj_step(self.model, d, nstep=4)
                mujoco.mj_forward(self.model, d)

    def observe(self):
        """Return 42-state and a world-fixed local patch (anchor xyz/yaw + 9x9 heights)."""
        import mujoco
        if self.backend == "mjlab":
            return gpu_observation(self.scene,self.sim,self.qadr,self.vadr,self.n)
        else:
            rows, touches, grids = [], [], []
            for d in self.data:
                rows.append(np.concatenate((d.qpos[:7], d.qpos[self.qadr], d.qvel[:6], d.qvel[self.vadr])))
                active = set()
                for c in d.contact:
                    if c.dist <= .002:
                        active.update((int(c.geom1), int(c.geom2)))
                touches.append([g in active for g in self.contact_geoms])
                w, x, y, z = d.qpos[3:7]
                yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                r = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                points = offsets().numpy() @ r.T + d.qpos[:2]
                h = []
                for xy in points:
                    start = np.array([*xy, d.qpos[2]+2.])
                    dist = mujoco.mj_ray(self.model, d, start, np.array([0., 0., -1.]), self.geom_groups,
                                         1, -1, np.array([-1], dtype=np.int32))
                    h.append(start[2]-dist if dist >= 0 else np.nan)
                grids.append(h)
            physical = torch.tensor(np.array(rows), dtype=torch.float32, device=self.device)
            found = torch.tensor(touches, dtype=torch.float32, device=self.device)
            heights = torch.tensor(np.array(grids), dtype=torch.float32, device=self.device)
            valid = torch.isfinite(heights)
        return pack_observation(physical,found,heights,valid)

    def curriculum(self, ids, up, down):
        if self.scene is not None and self.terrain != "flat":
            self.scene["terrain"].update_env_origins(ids, up, down)

    def close(self):
        pass


def pack_observation(physical, found, heights, valid):
    if not valid.all():
        raise RuntimeError("terrain rays missed: cannot train with fabricated ground heights")
    hq = heading_quat(physical[:, 3:7])
    yaw = 2*torch.atan2(hq[:, 3], hq[:, 0])
    context = torch.cat((physical[:, :2], heights[:, GRID*GRID//2:GRID*GRID//2+1], yaw[:, None], heights), -1)
    state = torch.cat((physical, found), -1).clone()
    if not torch.isfinite(state).all():
        raise FloatingPointError("non-finite velocity physics state")
    return state, context.clone()


def gpu_observation(scene, simulation, qadr, vadr, n):
    q, v = simulation.data.qpos[:], simulation.data.qvel[:]
    physical = torch.cat((q[:, :7], q[:, qadr], v[:, :6], v[:, vadr]), -1)
    found = torch.stack([scene[f"velocity_contact_{i}"].data.found.reshape(n,-1).any(-1)
                         for i in range(5)],-1).float()
    rays = scene["velocity_ground"].data
    heights = rays.hit_pos_w[...,2]
    valid = (rays.distances >= 0) & torch.isfinite(heights)
    return pack_observation(physical,found,heights,valid)
