from reskill.rl.envs import fetch_stack_env
from gym import utils, spaces
import numpy as np

DISTANCE_THRESHOLD = 0.04
DSL_DEBUG = True

# Custom Environments


class FetchStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(
        self, reward_info="stack_red", use_fixed_goal=False, use_force_sensor=True
    ):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object1:joint": [1.25, 0.53, 0.46, 1.0, 0.0, 0.0, 0.0],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self,
            "fetch/stack2.xml",
            num_blocks=2,
            block_gripper=False,
            n_substeps=50,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.10,
            target_range=0.10,
            distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos,
            reward_info=reward_info,
            goals_on_stack_probability=0.0,
            use_fixed_goal=use_fixed_goal,
            use_force_sensor=use_force_sensor,
        )  # 0.2
        utils.EzPickle.__init__(self)


class FetchCleanUpEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(
        self,
        seed=0,
        reward_info="cleanup_1block",
        use_fixed_goal=True,
        use_force_sensor=True,
    ):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self,
            "fetch/table_cleanup_1_block.xml",
            num_blocks=1,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos,
            reward_info=reward_info,
            use_fixed_goal=True,
            use_force_sensor=True,
            seed=seed,
        )  # 0.2
        utils.EzPickle.__init__(self, seed)


class FetchPlaceEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(
        self, seed, reward_info="place", use_fixed_goal=True, use_force_sensor=True
    ):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self,
            "fetch/stack1.xml",
            num_blocks=1,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos,
            reward_info=reward_info,
            use_fixed_goal=use_fixed_goal,
            use_force_sensor=True,
            seed=seed,
        )
        utils.EzPickle.__init__(self, seed)

    def get_obs(self, block_idx=0):
        obs = self._get_obs()
        # obs = self.obs
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
        obs = self.obs
        gripper_state = obs["observation"][3:5]
        return abs(gripper_state[0] - 0.024) < atol or gripper_state[0] - 0.024 < 0

    def gripper_are_open(self, atol=1e-3):
        obs = self.obs
        gripper_state = obs["observation"][3:5]
        # return abs(gripper_state[0] - 0.05) < atol
        return gripper_state[0] > 0.024 + atol


class FetchPlaceAbsEnv(FetchPlaceEnv):
    def __init__(
        self,
        seed,
        reward_info="place",
        use_fixed_goal=True,
        use_force_sensor=True,
        b_slippery=True,
        slippery_prob=0.2
    ):
        FetchPlaceEnv.__init__(
            self, seed, reward_info, use_fixed_goal, use_force_sensor
        )

        # move_to_block, move_to_goal, move_down, open_gripper, close_gripper
        self.action_space = spaces.Discrete(5)

        # block_at_goal, block_is_grasped, block_above_goal, block_below_gripper, block_inside_gripper, gripper_are_open
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6,))
        self.grasped = False
        self.b_slippery = b_slippery
        self.slippery_prob = slippery_prob

    def _get_abs_obs(self):
        block_at_goal = self.block_at_goal()
        block_is_grasped = self.block_is_grasped()
        block_above_goal = self.block_above_goal()
        block_below_gripper = self.block_below_gripper()
        block_inside_gripper = self.block_inside_gripper()
        gripper_open = self.gripper_are_open()

        observation = [
            float(block_at_goal),
            float(block_is_grasped),
            float(block_above_goal),
            float(block_below_gripper),
            float(block_inside_gripper),
            float(gripper_open),
        ]
        return np.array(observation)

    def reset(self):
        self.grasped = False
        FetchPlaceEnv.reset(self)
        return self._get_abs_obs()

    def step(self, action):
        b_slippery = (
            self.b_slippery
            and (np.random.rand() < self.slippery_prob)
            and self.block_is_grasped()
        )
        print("action is ", action)
        if action == 0:
            # move_to_block
            raw_obs = self.obs
            _, block_position, _ = self.get_obs()
            target_position = self.get_target_position(block_position)
            robot_action = self.get_move_action(raw_obs, target_position)
            if DSL_DEBUG:
                print("move to block")
        elif action == 1:
            # move_to_goal
            raw_obs = self.obs
            _, block_position, place_position = self.get_obs()
            target_position = self.get_target_position(
                place_position, workspace_height=0.02
            )
            robot_action = self.get_move_action(
                raw_obs, target_position, close_gripper=True
            )
            if DSL_DEBUG:
                print("move to goal")
        elif action == 2:
            # move down
            raw_obs = self.obs
            gripper_position, block_position, place_position = self.get_obs()
            relative_grasp_position = (0.0, 0.0, -0.02)
            target_position = np.add(block_position, relative_grasp_position)
            robot_action = self.get_move_action(raw_obs, target_position)
            if DSL_DEBUG:
                print("move doen")
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
        else:
            print("action is ", action)
        if b_slippery:
            print("block is made slippery")
            robot_action[-1] = 1.0
        obs, reward, done, info = FetchPlaceEnv.step(self, robot_action)

        if reward == 1.0:
            reward = 0.5
            done = True
        if self.block_is_grasped() and reward != 1 and not self.grasped:
            reward = 0.5
            self.grasped = True
            print("using grasp reward")
        print("reward is", reward)
        return self._get_abs_obs(), reward, done, info

    def get_target_position(
        self,
        place_position,
        relative_grasp_position=(0.0, 0.0, -0.02),
        workspace_height=0.1,
    ):
        target_position = np.add(place_position, relative_grasp_position)
        target_position[2] += workspace_height * 1
        return target_position

    def get_move_action(
        self, observation, target_position, atol=1e-3, gain=10.0, close_gripper=False
    ):
        """
        Move an end effector to a position and orientation.
        """
        # Get the currents
        current_position = observation["observation"][:3]

        action = gain * np.subtract(target_position, current_position)
        if close_gripper:
            gripper_action = -1.0
        else:
            gripper_action = 0.0
        action = np.hstack((action, gripper_action))

        return action


class FetchSlipperyPushEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info="place", use_fixed_goal=True):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self,
            "fetch/slippery_push.xml",
            num_blocks=1,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos,
            reward_info=reward_info,
            use_fixed_goal=use_fixed_goal,
        )
        utils.EzPickle.__init__(self)

        for i in range(len(self.sim.model.geom_friction)):
            self.sim.model.geom_friction[i] = [
                25e-2,
                5.0e-3,
                1e-4,
            ]  # [1e+00, 5.e-3, 1e-4]


class FetchBlockStackEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(
        self, reward_info="incremental", use_fixed_goal=True, use_force_sensor=True
    ):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self,
            "fetch/block_stacking.xml",
            num_blocks=1,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos,
            reward_info=reward_info,
            use_fixed_goal=use_fixed_goal,
            use_force_sensor=True,
        )
        utils.EzPickle.__init__(self)
