import os
import copy
import numpy as np

import gym
from gym import error, spaces

# from gym.utils import seeding
from mujoco_py import GlfwContext
import pdb

from reskill.rl.envs import rotations, robot_env

# import utils as ru

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )

# main a single mujoco model, information needed is (model_path, n_substeps)
model_path = "hook.xml"
n_substeps = 20
######### needed to be set before use

if model_path.startswith("/"):
    fullpath = model_path
else:
    fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
if not os.path.exists(fullpath):
    raise IOError("File {} does not exist".format(fullpath))

model = mujoco_py.load_model_from_path(fullpath)
sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
# viewer = mujoco_py.MjViewer(sim)
viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)


def _viewer_setup():
    body_id = sim.model.body_name2id("robot0:gripper_link")
    lookat = sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
        viewer.cam.lookat[idx] = value
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 180.0
    viewer.cam.elevation = -24.0


_viewer_setup()


class FetchHookOptimized(gym.Env):
    def __init__(self, seed):
        self._goal_pos = np.array([1.65, 0.75, 0.42])
        self._object_xpos = np.array([1.8, 0.75])

        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
            "hook:joint": [1.35, 0.35, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        # input argument
        n_actions = 4
        has_object = True
        block_gripper = False
        n_substeps = 20
        gripper_extra_height = 0.2
        target_in_the_air = True
        target_offset = 0.0
        obj_range = None
        target_range = None
        distance_threshold = 0.05
        reward_type = "sparse"

        # actual content of __init__ for gym env
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        self.n_substeps = n_substeps
        self.initial_qpos = initial_qpos

        sim.reset()
        self.seed(seed)
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(sim.get_state())
        # print(self.initial_state)
        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.obs = obs
        self._store_sim_state()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

    @property
    def dt(self):
        return sim.model.opt.timestep * sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        # if seed is None:
            # print("seed input is None")
        # else:
            # print("seed input is ", seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # print("seed output is ", seed)
        return [seed]

    def _set_sim_state(self):
        sim.set_state(self.sim_state)
        sim.forward()

    def _store_sim_state(self):
        self.sim_state = sim.get_state()

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self._set_sim_state()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        sim.step()

        self._step_callback()

        # if self.render_mode == "human":
        # self.render()
        self._store_sim_state()
        obs = self._get_obs()
        self.obs = obs

        done = False

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        return obs, reward, done, info

    def get_reward(self):
        self._set_sim_state()
        obs = self._get_obs()
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        reward = self.compute_reward(
            obs["achieved_goal"],
            self.goal,
            info,
        )
        return reward

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.

        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        self.obs = obs.copy()

        self._store_sim_state()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None

    def render(self, mode="human", width=3000, height=2000):
        self._render_callback()
        if mode == "rgb_array":
            viewer.render(width, height)
            data = viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]

        elif mode == "human":
            viewer.render()

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        sim.set_state(self.initial_state)

        object_xpos_x = 1.65 + self.np_random.uniform(-0.05, 0.05)
        while True:
            object_xpos_x = 1.8 + self.np_random.uniform(-0.05, 0.10)
            object_xpos_y = 0.75 + self.np_random.uniform(-0.05, 0.05)
            if (object_xpos_x - self._goal_pos[0]) ** 2 + (
                object_xpos_y - self._goal_pos[1]
            ) ** 2 >= 0.01:
                break
        self._object_xpos = np.array([object_xpos_x, object_xpos_y])

        object_qpos = sim.data.get_joint_qpos("object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = self._object_xpos
        sim.data.set_joint_qpos("object0:joint", object_qpos)

        sim.forward()
        return True

    def generate_mujoco_observations(self):
        # positions
        grip_pos = sim.data.get_site_xpos("robot0:grip")

        dt = sim.nsubsteps * sim.model.opt.timestep
        grip_velp = sim.data.get_site_xvelp("robot0:grip") * dt

        robot_qpos, robot_qvel = robot_get_obs(sim)
        if self.has_object:
            object_pos = sim.data.get_site_xpos("object0")
            # rotations
            object_rot = rotations.mat2euler(sim.data.get_site_xmat("object0"))
            # velocities
            object_velp = sim.data.get_site_xvelp("object0") * dt
            object_velr = sim.data.get_site_xvelr("object0") * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )

    def _base_fetch_get_obs(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _get_obs(self):
        obs = self._base_fetch_get_obs()

        grip_pos = sim.data.get_site_xpos("robot0:grip")
        dt = sim.nsubsteps * sim.model.opt.timestep
        grip_velp = sim.data.get_site_xvelp("robot0:grip") * dt

        hook_pos = sim.data.get_site_xpos("hook")
        # rotations
        hook_rot = rotations.mat2euler(sim.data.get_site_xmat("hook"))
        # velocities
        hook_velp = sim.data.get_site_xvelp("hook") * dt
        hook_velr = sim.data.get_site_xvelr("hook") * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate(
            [hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos]
        )

        obs["observation"] = np.concatenate([obs["observation"], hook_observation])

        return obs

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        # rot_ctrl = [ 0.5, -0.5, 0.5, 0.5 ]  # 90 deg rotation of the original end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        ctrl_set_action(sim, action)
        mocap_set_action(sim, action)

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self, return_extra_info=False):
        goals_pos = self._goal_pos.copy()
        return goals_pos

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            sim.data.set_joint_qpos(name, value)
        reset_mocap_welds(sim)
        sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + sim.data.get_site_xpos("robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)

        for _ in range(10):
            sim.step()

            # Extract information for sampling goals.
        self.initial_gripper_xpos = sim.data.get_site_xpos("robot0:grip").copy()
        self.height_offset = sim.data.get_site_xpos("object0")[2]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (sim.data.site_xpos - sim.model.site_pos).copy()
        site_id = sim.model.site_name2id("target0")
        sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        sim.forward()

    def _step_callback(self):
        pass

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return (d <= self.distance_threshold).astype(np.float32)
        else:
            return -d

    def get_obs(self):
        # obs = self._get_obs()
        obs = self.obs
        gripper_position = obs["observation"][:3]
        block_position = obs["observation"][3:6]
        hook_position = obs["observation"][25:28]
        place_position = obs["desired_goal"]
        return gripper_position, block_position, hook_position, place_position

    ############################
    def object_at_target(self, atol=1e-3):
        _, _, hook_position, _ = self.get_obs()
        target_position = hook_position.copy()
        target_position[2] = 0.5
        d = goal_distance(hook_position, target_position)
        return d <= self.distance_threshold

    def object_is_grasped(self, relative_grasp_position=(0.0, 0.0, -0.02), atol=1e-3):
        return self.object_inside_gripper(
            relative_grasp_position, atol
        ) and self.gripper_are_closed(atol)

    def object_below_gripper(self, atol=1e-3):
        gripper_position, _, hook_position, _ = self.get_obs()
        return (gripper_position[0] - hook_position[0]) ** 2 + (
            gripper_position[1] - hook_position[1]
        ) ** 2 < atol

    def object_inside_gripper(
        self, relative_grasp_position=(0.0, 0.0, -0.02), atol=1e-3
    ):
        # gripper_position, block_position, hook_position, place_position = self.get_obs()
        gripper_position, _, hook_position, _ = self.get_obs()
        relative_position = np.subtract(gripper_position, hook_position)

        return (
            np.sum(np.subtract(relative_position, relative_grasp_position) ** 2) < atol
        )

    def gripper_are_closed(self, atol=1e-3):
        threshold = 0.024
        obs = self.obs
        gripper_state = obs["observation"][9:11]
        return (
            abs(gripper_state[0] - threshold) < atol or gripper_state[0] - threshold < 0
        )

    def gripper_are_open(self, atol=1e-3):
        threshold = 0.024
        obs = self.obs
        gripper_state = obs["observation"][9:11]
        # return abs(gripper_state[0] - 0.05) < atol
        return gripper_state[0] > threshold + atol

    ###########################
    def block_at_goal(self, atol=0.03):
        _, block_position, _, place_position = self.get_obs()
        # return np.sum(np.subtract(block_position, place_position) ** 2) < atol
        return np.linalg.norm(np.subtract(block_position, place_position)) < atol

    def hook_aligned(self):
        _, block_position, hook_position, _ = self.get_obs()
        hook_target = np.array(
            [block_position[0] - 0.5, block_position[1] - 0.05, 0.45]
        )
        # print("hook position", hook_position)
        # print("hook target", hook_target)
        # print("hook target", hook_target)
        return (
            hook_position[0] < hook_target[0] + 0.1
            and hook_position[1] + 0.1 > hook_target[1]
        )

    def hook_grasped(self):
        gripper_position, _, hook_position, _ = self.get_obs()
        return is_grasped(
            self.obs,
            gripper_position=gripper_position,
            target_position=hook_position,
            relative_grasp_position=(0.0, 0.0, -0.05),
            atol=1e-2,
        )


def is_grasped(
    raw_obs, gripper_position, target_position, relative_grasp_position, atol=1e-3
):
    block_inside = is_inside_gripper(
        gripper_position, target_position, relative_grasp_position, atol=atol
    )
    grippers_closed = grippers_are_closed(raw_obs, atol=atol)
    return block_inside and grippers_closed


def is_inside_gripper(
    gripper_position, target_position, relative_grasp_position, atol=1e-3
):
    relative_position = np.subtract(gripper_position, target_position)
    return np.sum(np.subtract(relative_position, relative_grasp_position) ** 2) < atol


def grippers_are_closed(raw_obs, atol=1e-3):
    threshold = 0.024
    gripper_state = raw_obs["observation"][9:11]
    return abs(gripper_state[0] - threshold) < atol or gripper_state[0] - threshold < 0


def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith("robot")]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7,))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7,))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation."""
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (
        sim.model.eq_type is None
        or sim.model.eq_obj1id is None
        or sim.model.eq_obj2id is None
    ):
        return
    for eq_type, obj1_id, obj2_id in zip(
        sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id
    ):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert mocap_id != -1
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
