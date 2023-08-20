import numpy as np

DSL_DEBUG = False
g_counter = None
last_reward = None
last_success = None


class k_cond:
    """cond : cond_without_not
    | NOT C_LBRACE cond_without_not C_RBRACE
    """

    def __init__(self, negation: bool, cond):
        self.negation = negation
        self.cond = cond

    def __call__(self, k):
        if self.negation:
            return not self.cond(k)
        else:
            return self.cond(k)

    def __str__(self):
        if self.negation:
            # return "NOT c( " + str(self.cond) + " c)"
            return "not (" + str(self.cond) + ")"
        else:
            return str(self.cond)

    def __eq__(self, other_cond):
        return self.negation == other_cond.negation and self.cond == other_cond.cond


class k_cond_without_not:
    """cond_without_not : FRONT_IS_CLEAR
    | LEFT_IS_CLEAR
    | RIGHT_IS_CLEAR
    | NO_MARKERS_PRESENT
    | FRONT_IS_DOOR
    | LEFT_IS_GOAL
    | RIGHT_IS_GOAL
    | GOAL_ON_RIGHT
    """

    def __init__(self, cond: str):
        self.cond = cond

    def __call__(self, k):
        return getattr(k, self.cond)()

    def __str__(self):
        return str(self.cond)


class k_action:
    """action : MOVE
    | TURN_RIGHT
    | TURN_LEFT
    | PICK_MARKER
    | PUT_MARKER
    | TOGGLE
    | NULL
    """

    def __init__(self, action: str):
        self.action = action

    def __call__(self, k, robot=None):
        if self.action == "null":
            return robot.check_reward()


        # g_op = k.gripper_are_open()
        # g_bk = k.block_inside_gripper()

        if not robot.no_fuel():
            # getattr(k, self.action)()
            if self.action == "move_to_push_position":
                raw_obs = k.env.obs
                push_position, _, _, _ = k.env.compute_push_position_stat()
                target_position = push_position.copy()
                workspace_height = 0.1
                target_position[2] += workspace_height
                action = get_move_action(raw_obs, target_position, 1e-4)
                # pdb.set_trace()
                # k.render()
                if DSL_DEBUG:
                    print("move to push position")
                _, rwd, done, info = k.step(action)
                # k.render()
            elif self.action == "move_down":
                raw_obs = k.env.obs
                push_position, _, _, _ = k.env.compute_push_position_stat()
                action = get_move_action(raw_obs, push_position, atol=1e-4)
                # pdb.set_trace()
                # k.render()
                if DSL_DEBUG:
                    print("move down")
                _, rwd, done, info = k.step(action)
                # k.render()
            elif self.action == "move_to_goal":
                raw_obs = k.env.obs
                gripper_position, block_position, goal = k.get_obs()
                target_position = goal.copy()
                target_position[2] = gripper_position[2]
                action = get_move_action(raw_obs, target_position, atol=1e-4, gain=5.0)
                # pdb.set_trace()
                # k.render()
                if DSL_DEBUG:
                    print("move to goal")
                _, rwd, done, info = k.step(action)
                # k.render()
            elif self.action == "idle":
                rwd = 0
                k._elapsed_steps += 1
                if DSL_DEBUG:
                    print("idle action")
            # import pdb
            # pdb.set_trace()
            if self.action != "idle":
                if DSL_DEBUG:
                    # pass
                    # data = k.render(mode="rgb_array")
                    k.render(mode="human")
                global g_counter
                global last_reward
                global last_success
                last_reward = rwd
                success = info["is_success"]
                last_success = success
                if g_counter is None:
                    g_counter = 0
                else:
                    g_counter += 1

                if DSL_DEBUG:
                    print("g_counter", g_counter, "_elapsed_steps", k._elapsed_steps)
                    print("done", done)
                    print("info", info)
                    print("type(k)", type(k))
                    print(f"(rwd, success): ({rwd}, {success})")
                    # cv2.imwrite(f"frames/img{g_counter:06d}.png", data)
                    print("saving rendered image")
            else:
                rwd = last_reward
                success = last_success
            if DSL_DEBUG:
                # pass
                print("block_at_goal", k.block_at_goal())
                print("block_is_grasped", k.block_is_grasped())
                print("block_above_goal", k.block_above_goal())
                print("block_inside_gripper", k.block_inside_gripper())
                print("block_below_gripper", k.block_below_gripper())
                print("gripper_are_closed", k.gripper_are_closed())
                print("gripper_are_open", k.gripper_are_open())
                # print("rwd is ", rwd)
            if k.block_at_goal():
                rwd = 1.0
            # elif k.block_is_grasped():
            # rwd = 0.5
            # the gripper is openned unexpectedly == bad behavior?
            # elif (
            #     not g_op
            #     and k.gripper_are_open()
            #     and not (self.action == "open_gripper")
            # ):
            #     print(f"action {self.action} make the gripper open unexpectedly")
            #     rwd = -1.0
            else:
                rwd = 0.0
            return rwd, success
        else:
            return 0.0, False

    def __str__(self):
        return " " + str(self.action)


class k_if:
    """if : IF C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE"""

    def __init__(self):
        # self.cond = None
        self.abs_state = []  # CNF, list of conds
        self.stmts = []

    # def __call__(self, k, robot=None):
    #     if not robot.no_fuel:
    #         if self.cond(k):
    #             for s in self.stmts:
    #                 s(k, robot)

    def __call__(self, k, robot=None):
        if not robot.no_fuel:
            result = True
            for cond in self.abs_state:
                if not cond(k):
                    result = False
                    break
            if result:
                for s in self.stmts:
                    s(k, robot)

    # def __str__(self):
    #     return "IF c( " + str(self.cond) + " c) i( " + str(self.stmts) + " i)"

    def __str__(self):
        return "TODO"


def get_move_action(
    observation, target_position, atol=1e-3, gain=10.0, close_gripper=False
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


def get_target_position(
    place_position, relative_grasp_position=(0.0, 0.0, -0.02), workspace_height=0.1
):
    target_position = np.add(place_position, relative_grasp_position)
    target_position[2] += workspace_height * 1
    return target_position


def get_target_position_without_height(
    place_position, relative_grasp_position=(0.0, 0.0, -0.02), workspace_height=0.1
):
    target_position = np.add(place_position, relative_grasp_position)
    return target_position
