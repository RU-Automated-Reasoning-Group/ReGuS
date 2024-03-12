from gymnasium_robotics.utils import rotations
from gymnasium_robotics.envs.fetch import fetch_env
from gymnasium import utils, spaces
import numpy as np
import os
import pdb
import time

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DSL_DEBUG = False


class FetchHookEnv(fetch_env.MujocoPyFetchEnv, utils.EzPickle):
    def __init__(self, xml_file=None, render_mode=None):
        # pdb.set_trace()
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
            "hook:joint": [1.35, 0.35, 0.4, 1.0, 0.0, 0.0, 0.0],
        }

        if xml_file is None:
            xml_file = os.path.join(DIR_PATH, "assets", "hook.xml")

        # pdb.set_trace()
        self._goal_pos = np.array([1.65, 0.75, 0.42])
        self._object_xpos = np.array([1.8, 0.75])

        fetch_env.MujocoPyFetchEnv.__init__(
            self,
            model_path=xml_file,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=None,
            target_range=None,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type="sparse",
        )

        utils.EzPickle.__init__(self)
        self.render_mode = render_mode

    def render(self, mode="rgb_array", *args, **kwargs):
        # See https://github.com/openai/gym/issues/1081
        self._render_callback()
        # pdb.set_trace()
        if mode == "rgb_array":
            # self._get_viewer(mode).render()
            width, height = 3000, 2000
            self._get_viewer(mode).render(width, height)
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

        return super(FetchHookEnv, self).render(*args, **kwargs)

    def _sample_goal(self):
        goal_pos = self._goal_pos.copy()
        # goal_pos[:2] += self.np_random.uniform(-0.05, 0.05)
        return goal_pos

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 180.0
        self.viewer.cam.elevation = -24.0

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        object_xpos_x = 1.65 + self.np_random.uniform(-0.05, 0.05)
        while True:
            object_xpos_x = 1.8 + self.np_random.uniform(-0.05, 0.10)
            object_xpos_y = 0.75 + self.np_random.uniform(-0.05, 0.05)
            if (object_xpos_x - self._goal_pos[0]) ** 2 + (
                object_xpos_y - self._goal_pos[1]
            ) ** 2 >= 0.01:
                break
        self._object_xpos = np.array([object_xpos_x, object_xpos_y])

        object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = self._object_xpos
        self.sim.data.set_joint_qpos("object0:joint", object_qpos)

        self.sim.forward()
        return True

    def _get_obs(self):
        obs = fetch_env.MujocoPyFetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt

        hook_pos = self.sim.data.get_site_xpos("hook")
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat("hook"))
        # velocities
        hook_velp = self.sim.data.get_site_xvelp("hook") * dt
        hook_velr = self.sim.data.get_site_xvelr("hook") * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate(
            [hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos]
        )

        obs["observation"] = np.concatenate([obs["observation"], hook_observation])

        return obs

    def _noisify_obs(self, obs, noise=1.0):
        return obs + np.random.normal(0, noise, size=obs.shape)


class FetchHookAbsEnv(FetchHookEnv):
    def __init__(self, xml_file=None, render_mode=None):
        FetchHookEnv.__init__(self, xml_file, render_mode)

        # align, sweep, move_to_object, open_gripper, close_gripper, move_down, move_to_target
        # self.action_space = spaces.Discrete(7)

        # # block_at_goal, hood_grasped, hook_aligned, object_at_target, object_below_gripper, gripper_are_open, object_inside_gripper, object_is_grasped
        # self.observation_space = spaces.Dict(
        #     dict(
        #         desired_goal=spaces.Box(
        #             -np.inf, np.inf, shape=(1,), dtype="float64"
        #         ),
        #         achieved_goal=spaces.Box(
        #             -np.inf, np.inf, shape=(1,), dtype="float64"
        #         ),
        #         observation=spaces.Box(0.0, 1.0, shape=(8,))
        #     )
        # )
        
        self.observation_space = spaces.Box(0.0, 1.0, shape=(8,))

    def public_get_abs_obs(self):
        return self._get_abs_obs()

    def _get_abs_obs(self):
        obs = self._get_obs()
        (
            gripper_position,
            block_position,
            hook_position,
            place_position,
            gripper_state,
        ) = self.parse_obs(obs)

        b_object_at_target = self.object_at_target(hook_position)
        b_object_below_gripper = self.object_below_gripper(
            gripper_position, hook_position
        )
        b_gripper_is_open = self.gripper_is_open(gripper_state)
        b_object_inside_gripper = self.object_insider_gripper(
            gripper_position, hook_position
        )
        b_object_is_grasped = self.object_is_grasped(
            gripper_position, hook_position, gripper_state
        )
        b_block_at_goal = self.block_at_goal(block_position, place_position)
        b_hook_grasped = self.hook_grasped(
            gripper_position, hook_position, gripper_state
        )
        b_hook_aligned = self.hook_aligned(block_position, hook_position)

        observation = [
            float(b_object_at_target),
            float(b_object_below_gripper),
            float(b_gripper_is_open),
            float(b_object_inside_gripper),
            float(b_object_is_grasped),
            float(b_block_at_goal),
            float(b_hook_grasped),
            float(b_hook_aligned),
        ]
        # print(observation)
        return np.array(observation)

    def object_at_target(self, hook_position):
        target_position = hook_position.copy()
        target_position[2] = 0.5
        d = self.goal_distance(hook_position, target_position)
        return d <= self.distance_threshold

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def object_below_gripper(self, gripper_position, hook_position, atol=1e-3):
        return (gripper_position[0] - hook_position[0]) ** 2 + (
            gripper_position[1] - hook_position[1]
        ) ** 2 < atol

    def gripper_is_open(self, gripper_state, atol=1e-3):
        threshold = 0.024
        return gripper_state[0] > threshold + atol

    def gripper_is_closed(self, gripper_state, atol=1e-3):
        threshold = 0.024
        return (
            abs(gripper_state[0] - threshold) < atol or gripper_state[0] - threshold < 0
        )

    def object_insider_gripper(
        self,
        gripper_position,
        hook_position,
        relative_grasp_position=(0.0, 0.0, -0.02),
        atol=1e-3,
    ):
        relative_position = np.subtract(gripper_position, hook_position)

        return (
            np.sum(np.subtract(relative_position, relative_grasp_position) ** 2) < atol
        )

    def object_is_grasped(
        self,
        gripper_position,
        hook_position,
        gripper_state,
        relative_grasp_position=(0.0, 0.0, -0.02),
        atol=1e-3,
    ):
        return self.object_insider_gripper(
            gripper_position, hook_position, relative_grasp_position, atol
        ) and self.gripper_is_closed(gripper_state, atol)

    def block_at_goal(self, block_position, place_position, atol=0.03):
        return np.linalg.norm(np.subtract(block_position, place_position)) < atol

    def hook_aligned(self, block_position, hook_position):
        hook_target = np.array(
            [block_position[0] - 0.5, block_position[1] - 0.05, 0.45]
        )
        return (
            hook_position[0] < hook_target[0] + 0.1
            and hook_position[1] + 0.1 > hook_target[1]
        )

    def hook_grasped(self, gripper_position, hook_position, gripper_state):
        return self.is_grasped(
            gripper_position,
            hook_position,
            gripper_state,
            relative_grasp_position=(0.0, 0.0, -0.05),
            atol=1e-2,
        )

    def is_grasped(
        self,
        gripper_position,
        target_position,
        gripper_state,
        relative_grasp_position,
        atol=1e-3,
    ):
        block_inside = self.is_inside_gripper(
            gripper_position, target_position, relative_grasp_position, atol
        )
        gripper_closed = self.gripper_is_closed(gripper_state, atol)
        return block_inside and gripper_closed

    def is_inside_gripper(
        self, gripper_position, target_position, relative_grasp_position, atol=1e-3
    ):
        relative_position = np.subtract(gripper_position, target_position)
        return (
            np.sum(np.subtract(relative_position, relative_grasp_position) ** 2) < atol
        )

    def reset(self):
        FetchHookEnv.reset(self)
        return self._get_abs_obs()

    def parse_obs(self, obs):
        gripper_position = obs["observation"][:3]
        block_position = obs["observation"][3:6]
        hook_position = obs["observation"][25:28]
        place_position = obs["desired_goal"]
        gripper_state = obs["observation"][9:11]
        return (
            gripper_position,
            block_position,
            hook_position,
            place_position,
            gripper_state,
        )

    def get_move_action(
        self,
        gripper_position,
        target_position,
        atol=1e-3,
        gain=10.0,
        close_gripper=False,
    ):
        action = gain * np.subtract(target_position, gripper_position)
        if close_gripper:
            gripper_action = -1.0
        else:
            gripper_action = 0.0
        action = np.hstack((action, gripper_action))

        return action

    def get_target_position(
        self,
        place_position,
        relative_grasp_position=(0.0, 0.0, -0.02),
        workspace_height=0.1,
    ):
        target_position = np.add(place_position, relative_grasp_position)
        target_position[2] += workspace_height * 1
        return target_position

    def step(self, action):
        obs = self._get_obs()
        (
            gripper_position,
            block_position,
            hook_position,
            place_position,
            gripper_state,
        ) = self.parse_obs(obs)
        print("action is ", action)
        if action == 0:
            # move_to_hook
            target_position = self.get_target_position(hook_position)
            robot_action = self.get_move_action(gripper_position, target_position)
            if DSL_DEBUG:
                print("move to hook")
        elif action == 1:
            # move_to_hook_goal
            target_position = hook_position.copy()
            target_position[2] = 0.5
            robot_action = self.get_move_action(
                gripper_position, target_position, close_gripper=True
            )
            if DSL_DEBUG:
                print("move to hook goal")
        elif action == 2:
            # move down
            target_position = self.get_target_position(
                hook_position,
                relative_grasp_position=(0.0, 0.0, -0.05),
                workspace_height=0.0,
            )
            robot_action = self.get_move_action(gripper_position, target_position)
            if DSL_DEBUG:
                print("move down")
        elif action == 3:
            # open gripper
            robot_action = np.array([0.0, 0.0, 0.0, 1.0])
            if DSL_DEBUG:
                print("open gripper")
        elif action == 4:
            # close gripper
            robot_action = np.array([0.0, 0.0, 0.0, -1.0])
            if DSL_DEBUG:
                print("close gripper")
        elif action == 5:
            # align
            hook_target = np.array(
                [block_position[0] - 0.5, block_position[1] - 0.05, 0.45]
            )
            robot_action = self.get_move_action(
                gripper_position, hook_target, close_gripper=True
            )
        elif action == 6:
            # sweep
            direction = np.subtract(place_position, block_position)
            direction = direction[:2] / np.linalg.norm(direction[:2])
            # print("[DIRECTION]", direction)
            robot_action = np.array(
                # [2.0 * direction[0], 2.0 * direction[1], 2.0 * direction[2], -1.0]
                [3.0 * direction[0], 3.0 * direction[1], 0.0, -1.0]
            )
        else:
            print("action is ", action)
            assert False
        # pdb.set_trace()
        time.sleep(0.002)
        print(f"robot action is {robot_action}")
        obs, reward, terminated, truncated, info = FetchHookEnv.step(self, robot_action)

        # print("here")
        # pdb.set_trace()
        if reward == -1.0:
            reward = 0.0
        elif reward == 0.0:
            reward = 1.0
        else:
            assert False

        if reward == 1.0:
            truncated = True
        print("reward is", reward)
        return self._get_abs_obs(), reward, terminated, truncated, info


class NoisyFetchHookEnv(FetchHookEnv):
    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt

        hook_pos = self.sim.data.get_site_xpos("hook")
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat("hook"))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp("hook") * dt
        hook_velr = self.sim.data.get_site_xvelr("hook") * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate(
            [hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos]
        )

        obs["observation"] = np.concatenate([obs["observation"], hook_observation])
        obs["observation"][3:5] = self._noisify_obs(
            obs["observation"][3:5], noise=0.025
        )
        obs["observation"][6:9] = (
            obs["observation"][3:6] - obs["observation"][:3]
        )  # object_pos - grip_pos
        obs["observation"][12:15] = self._noisify_obs(
            obs["observation"][6:9], noise=0.025
        )
        return obs

    def _noisify_obs(self, obs, noise=1.0):
        return obs + np.random.normal(0, noise, size=obs.shape)

    def compute_reward(self, *args, **kwargs):
        return FetchHookEnv.compute_reward(self, *args, **kwargs)

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation

        return observation


class TwoFrameHookNoisyEnv(FetchHookEnv):
    def __init__(self, xml_file=None):
        super(TwoFrameHookNoisyEnv, self).__init__()
        self.observation_space.spaces["observation"] = spaces.Box(
            low=np.hstack(
                (
                    self.observation_space.spaces["observation"].low,
                    self.observation_space.spaces["observation"].low,
                )
            ),
            high=np.hstack(
                (
                    self.observation_space.spaces["observation"].high,
                    self.observation_space.spaces["observation"].high,
                )
            ),
            dtype=np.float32,
        )

    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt

        hook_pos = self.sim.data.get_site_xpos("hook")
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat("hook"))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp("hook") * dt
        hook_velr = self.sim.data.get_site_xvelr("hook") * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate(
            [hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos]
        )

        obs["observation"] = np.concatenate([obs["observation"], hook_observation])
        obs["observation"][3:5] = self._noisify_obs(
            obs["observation"][3:5], noise=0.025
        )
        obs["observation"][6:9] = (
            obs["observation"][3:6] - obs["observation"][:3]
        )  # object_pos - grip_pos
        obs["observation"][12:15] = self._noisify_obs(
            obs["observation"][6:9], noise=0.025
        )
        return obs

    def step(self, action):
        observation, reward, done, debug_info = FetchHookEnv.step(self, action)

        obs_out = observation.copy()
        obs_out["observation"] = np.hstack(
            (self._last_observation["observation"], observation["observation"])
        )
        self._last_observation = observation

        return obs_out, reward, done, debug_info

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation.copy()
        observation["observation"] = np.hstack(
            (self._last_observation["observation"], observation["observation"])
        )
        return observation

    def _noisify_obs(self, obs, noise=1.0):
        return obs + np.random.normal(0, noise, size=obs.shape)
