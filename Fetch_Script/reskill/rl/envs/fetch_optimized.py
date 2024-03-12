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
model_path = "fetch/stack1.xml"
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
    # lookat = [1.35268506, 0.74301371, 0.4008681]
    lookat = [1.27998563, 0.68635066, 0.35350562]

    for idx in range(3):
        # on_screen_viewer.cam.lookat[idx] = lookat[idx]
        viewer.cam.lookat[idx] = lookat[idx]
    # self.viewer.cam.distance = 0.8420461999474638 #2.5
    # self.viewer.cam.azimuth = 42.48803827751195 #132
    # self.viewer.cam.elevation = -22.612440191387563 #-14
    # on_screen_viewer.cam.distance = 0.8547035766991275
    # on_screen_viewer.cam.azimuth = 124.95215311004816
    # on_screen_viewer.cam.elevation = -22.488038277512022

    viewer.cam.distance = 0.8547035766991275
    viewer.cam.azimuth = 124.95215311004816
    viewer.cam.elevation = -22.488038277512022


_viewer_setup()


class FetchOptimizedEnv(gym.Env):
    def __init__(self, seed):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        n_actions = 4
        num_blocks = 1
        gripper_extra_height = 0.2
        block_gripper = False
        target_in_the_air = False
        target_offset = 0.0
        obj_range = 0.15
        target_range = 0.15
        DISTANCE_THRESHOLD = 0.04
        distance_threshold = DISTANCE_THRESHOLD
        reward_info = "place"
        use_fixed_goal = False
        use_force_sensor = True
        allow_blocks_on_stack = True
        goals_on_stack_probability = 1.0
        allow_blocks_on_stack = True
        all_goals_always_on_stack = False

        # fetch stack env
        self.num_blocks = num_blocks
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_info = reward_info
        self.goals_on_stack_probability = goals_on_stack_probability
        self.allow_blocks_on_stack = allow_blocks_on_stack
        self.all_goals_always_on_stack = all_goals_always_on_stack
        self.use_fixed_goal = use_fixed_goal
        self.use_force_sensor = use_force_sensor
        self.position = []

        self.object_names = ["object{}".format(i) for i in range(self.num_blocks)]

        self.location_record = None
        self.location_record_write_dir = None
        self.location_record_prefix = None
        self.location_record_file_number = 0
        self.location_record_steps_recorded = 0
        self.location_record_max_steps = 2000
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }
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
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
                force_sensor=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
            )
        )

    @property
    def dt(self):
        return sim.model.opt.timestep * sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        if seed is None:
            print("seed input is None")
        else:
            print("seed input is ", seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        print("seed output is ", seed)
        return [seed]

    def _set_sim_state(self):
        sim.set_state(self.sim_state)
        sim.forward()

    def _store_sim_state(self):
        self.sim_state = sim.get_state()

    def step(self, action):
        # set the sim state for current robot
        self._set_sim_state()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        sim.step()
        self._step_callback()

        # store the sim state
        self._store_sim_state()
        obs = self._get_obs()
        self.obs = obs

        done = False

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        # info = {}
        reward = self.compute_reward(
            obs["achieved_goal"],
            self.goal,
            obs["force_sensor"][2],
            obs["observation"][:3],
            obs["observation"][3:5],
            obs,
            info,
        )
        # print("Sim State", sim.get_state())
        # print("Stored state", sim.get_state() == self.sim_state)
        # print("box position", self.obs["achieved_goal"])
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
            obs["force_sensor"][2],
            obs["observation"][:3],
            obs["observation"][3:5],
            obs,
            info,
        )
        return reward

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.

        self.goal, goals, number_of_goals_along_stack = self._sample_goal(
            return_extra_info=True
        )

        if number_of_goals_along_stack == 0 or not self.allow_blocks_on_stack:
            number_of_blocks_along_stack = 0
        elif number_of_goals_along_stack < self.num_blocks:
            number_of_blocks_along_stack = np.random.randint(
                0, number_of_goals_along_stack + 1
            )
        else:
            number_of_blocks_along_stack = np.random.randint(
                0, number_of_goals_along_stack
            )

        # TODO remove line
        # number_of_blocks_along_stack = 0

        # print("number_of_goals_along_stack: {} number_of_blocks_along_stack: {}".format(number_of_goals_along_stack, number_of_blocks_along_stack))

        sim.set_state(self.initial_state)

        # Randomize start position of object.
        # prev_x_positions = [goal[:2] for goal in goals]  # Avoids blocks randomly being in goals
        prev_x_positions = [goals[0][:2]]
        for i, obj_name in enumerate(self.object_names):
            object_qpos = sim.data.get_joint_qpos("{}:joint".format(obj_name))
            assert object_qpos.shape == (7,)
            object_qpos[2] = 0.425  # 0.425

            # add noise to angle info
            # object_qpos[3] = np.random.normal(loc=0, scale=0.002, size=1)
            # object_qpos[4] = np.random.normal(loc=0, scale=0.002, size=1)
            # object_qpos[5] = np.random.normal(loc=0, scale=0.002, size=1)

            if i < number_of_blocks_along_stack:
                object_qpos[:3] = goals[i]
                object_qpos[:2] += np.random.normal(loc=0, scale=0.002, size=2)
            else:
                object_xpos = self.initial_gripper_xpos[:2].copy()

                while np.linalg.norm(
                    object_xpos - self.initial_gripper_xpos[:2]
                ) < 0.1 or np.any(
                    [
                        np.linalg.norm(object_xpos - other_xpos) < 0.05
                        for other_xpos in prev_x_positions
                    ]
                ):
                    object_xpos = self.initial_gripper_xpos[:2] + (
                        self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                        - 0.05
                    )  # -0.05) # TODO FOR THE CLEANUP ENV
                object_qpos[:2] = object_xpos

            prev_x_positions.append(object_qpos[:2])
            print("initial box position is ", object_qpos[:2])
            sim.data.set_joint_qpos("{}:joint".format(obj_name), object_qpos)

        sim.forward()

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
        sim.forward()
        return True

    def _get_obs(self):
        # positions
        grip_pos = sim.data.get_site_xpos("robot0:grip")
        dt = sim.nsubsteps * sim.model.opt.timestep
        grip_velp = sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = robot_get_obs(sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate(
            [
                grip_pos,
                gripper_state,
                grip_velp,
                gripper_vel,
            ]
        )

        achieved_goal = []

        for i in range(self.num_blocks):
            # for i in range(1):

            object_i_pos = sim.data.get_site_xpos(self.object_names[i])
            # rotations
            object_i_rot = rotations.mat2euler(
                sim.data.get_site_xmat(self.object_names[i])
            )
            # velocities
            object_i_velp = sim.data.get_site_xvelp(self.object_names[i]) * dt
            object_i_velr = sim.data.get_site_xvelr(self.object_names[i]) * dt
            # gripper state
            object_i_rel_pos = object_i_pos - grip_pos
            object_i_velp -= grip_velp

            obs = np.concatenate(
                [
                    obs,
                    object_i_pos.ravel(),
                    object_i_rel_pos.ravel(),
                    # object_i_rot.ravel(),
                    object_i_velp.ravel(),
                    # object_i_velr.ravel()
                ]
            )

            # This is current location of the blocks
            achieved_goal = np.concatenate([achieved_goal, object_i_pos.copy()])

        achieved_goal = np.concatenate([achieved_goal, grip_pos.copy()])

        achieved_goal = np.squeeze(achieved_goal)

        if self.use_force_sensor:
            sim.data.get_sensor("force_sensor")
            force_reading = sim.data.sensordata  # Read force sensor reading from tray
        else:
            force_reading = [0, 0, 0]

        # achieved_goal = np.squeeze(np.concatenate((object0_pos.copy(), object1_pos.copy())))
        #
        # obs = np.concatenate([
        #     grip_pos,
        #     object0_pos.ravel(), object1_pos.ravel(),
        #     object0_rel_pos.ravel(), object1_rel_pos.ravel(),
        #     gripper_state,
        #     object0_rot.ravel(), object1_rot.ravel(),
        #     object0_velp.ravel(), object1_velp.ravel(),
        #     object0_velr.ravel(), object1_velr.ravel(),
        #     grip_velp,
        #     gripper_vel,
        # ])

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
            "force_sensor": force_reading.copy(),
        }

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
        distances = self.sub_goal_distances(achieved_goal, desired_goal)
        if (
            sum([-(d > self.distance_threshold).astype(np.float32) for d in distances])
            == 0
        ):
            return True
        else:
            return False

    def sub_goal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        goal_a = goal_a[..., :-3]
        goal_b = goal_b[..., :-3]
        for i in range(self.num_blocks - 1):
            assert (
                goal_a[..., i * 3 : (i + 1) * 3].shape
                == goal_a[..., (i + 1) * 3 : (i + 2) * 3].shape
            )

        return [
            np.linalg.norm(
                goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3],
                axis=-1,
            )
            for i in range(self.num_blocks)
        ]

    def _sample_goal(self, return_extra_info=False):
        max_goals_along_stack = self.num_blocks
        # TODO was 2
        if self.all_goals_always_on_stack:
            min_goals_along_stack = self.num_blocks
        else:
            min_goals_along_stack = 1

        if np.random.uniform() < 1.0 - self.goals_on_stack_probability:
            max_goals_along_stack = 0
            min_goals_along_stack = 0

        number_of_goals_along_stack = np.random.randint(
            min_goals_along_stack, max_goals_along_stack + 1
        )

        goal0 = None
        first_goal_is_valid = False
        while not first_goal_is_valid:
            goal0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            if self.num_blocks > 4:
                if np.linalg.norm(goal0[:2] - self.initial_gripper_xpos[:2]) < 0.09:
                    continue
            first_goal_is_valid = True

        # goal0[0] = goal0[0] - 0.05
        goal0 += self.target_offset
        goal0[2] = self.height_offset
        goal0[1] += self.np_random.uniform(-0.35, 0.35, size=1)

        goals = [goal0]

        prev_x_positions = [goal0[:2]]
        goal_in_air_used = False
        for i in range(self.num_blocks - 1):
            if i < number_of_goals_along_stack - 1:
                goal_i = goal0.copy()
                goal_i[2] = self.height_offset + (0.05 * (i + 1))
            else:
                goal_i_set = False
                goal_i = None
                while not goal_i_set or np.any(
                    [
                        np.linalg.norm(goal_i[:2] - other_xpos) < 0.06
                        for other_xpos in prev_x_positions
                    ]
                ):
                    goal_i = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                        -self.target_range, self.target_range, size=3
                    )
                    goal_i_set = True

                goal_i += self.target_offset
                goal_i[2] = self.height_offset

            prev_x_positions.append(goal_i[:2])
            goals.append(goal_i)
        goals.append([0.0, 0.0, 0.0])

        if self.use_fixed_goal:
            # 1 block env
            if self.num_blocks == 1:
                if self.reward_info == "stack_red":
                    goals = [[1.416193226, 0.9074910037, 0.4245288], [0.0, 0.0, 0.0]]
                elif self.reward_info == "cleanup_1block":
                    goals = [[1.416193226, 1.074910037, 0.4245288], [0.0, 0.0, 0.0]]
                else:
                    goals = [[1.316193226, 0.7074910037, 0.4245288], [0.0, 0.0, 0.0]]

            elif self.num_blocks == 2:  # 2 block env
                goals = [[1.4, 1.06, 0.42], [1.40, 1.06, 0.42], [0.0, 0.0, 0.0]]

        if not return_extra_info:
            return np.concatenate(goals, axis=0).copy()
        else:
            print(f"sampled goals are {goals}")
            return (
                np.concatenate(goals, axis=0).copy(),
                goals,
                number_of_goals_along_stack,
            )

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            sim.data.set_joint_qpos(name, value)
        reset_mocap_welds(sim)
        # sim.forward()

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
        # append position
        gripper_id = sim.model.site_name2id("robot0:grip")
        self.position.append(sim.data.geom_xpos[gripper_id].copy())

        # Visualize target.
        sites_offset = (sim.data.site_xpos - sim.model.site_pos).copy()
        # print("sites offset: {}".format(sites_offset[0]))
        for i in range(self.num_blocks):
            site_id = sim.model.site_name2id("target{}".format(i))
            sim.model.site_pos[site_id] = (
                self.goal[i * 3 : (i + 1) * 3] - sites_offset[i]
            )

        # Visualise gripper position trajectory
        # if len(self.position) > 1:
        #     for i in range(len(self.position)-1):
        #         self.viewer.add_marker(pos=self.position[i], type=2, size=np.array([.005, .005, .005]), rgba=np.array([0, 1, 0, 0.05]), label="")

        sim.forward()

    def _step_callback(self):
        pass

    def compute_reward(
        self, achieved_goal, goal, force_sensor, gripper_pos, gripper_state, obs, info
    ):
        # Compute distance between goal and the achieved goal.
        # print(self.reward_info)
        # print("code in library")
        dist_b = np.linalg.norm(achieved_goal[0:2] - goal[0:2], axis=-1)
        print(f"dist_b {dist_b}")
        if dist_b < 0.03:
            return 1.0
        else:
            return 0.0

    def get_obs(self, block_idx=0):
        # obs = self._get_obs()
        obs = self.obs
        gripper_position = obs["observation"][:3]
        b_idx = block_idx * 9 + 10
        block_position = obs["observation"][b_idx : b_idx + 3]
        g_idx = block_idx * 3
        place_position = obs["desired_goal"][:3]
        return gripper_position, block_position, place_position

    def block_at_goal(self, atol=1e-3):
        gripper_position, block_position, place_position = self.get_obs()
        return np.sum(np.subtract(block_position, place_position) ** 2) < atol

    def block_is_grasped(self, relative_grasp_position=(0.0, 0.0, -0.02), atol=1e-3):
        return self.block_inside_gripper(
            relative_grasp_position, atol
        ) and self.gripper_are_closed(atol)

    def block_above_goal(
        self, relative_grasp_position=(0.0, 0.0, -0.02), workspace_height=0.1, atol=1e-3
    ):
        gripper_position, block_position, place_position = self.get_obs()
        target_position = np.add(place_position, relative_grasp_position)
        target_position[2] += workspace_height * 1
        return np.sum(np.subtract(block_position, target_position) ** 2) < atol

    def block_below_gripper(self, atol=1e-3):
        gripper_position, block_position, place_position = self.get_obs()
        return (gripper_position[0] - block_position[0]) ** 2 + (
            gripper_position[1] - block_position[1]
        ) ** 2 < atol

    def block_inside_gripper(
        self, relative_grasp_position=(0.0, 0.0, -0.02), atol=1e-3
    ):
        gripper_position, block_position, place_position = self.get_obs()
        relative_position = np.subtract(gripper_position, block_position)

        return (
            np.sum(np.subtract(relative_position, relative_grasp_position) ** 2) < atol
        )

    def gripper_are_closed(self, atol=1e-3):
        threshold = 0.024
        obs = self.obs
        gripper_state = obs["observation"][3:5]
        return abs(gripper_state[0] - threshold) < atol or gripper_state[0] - threshold < 0

    def gripper_are_open(self, atol=1e-3):
        threshold = 0.024
        obs = self.obs
        gripper_state = obs["observation"][3:5]
        # return abs(gripper_state[0] - 0.05) < atol
        return gripper_state[0] > threshold + atol


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
