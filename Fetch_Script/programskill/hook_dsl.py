import numpy as np
import cv2

DSL_DEBUG = False
g_counter = None
b_count = True
last_reward = None
last_success = None

def start_count():
    global b_count
    b_count = True

def stop_count():
    global b_count
    b_count = False

def get_g_counter():
    return g_counter

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

        import pdb

        # g_op = k.gripper_are_open()
        # g_bk = k.block_inside_gripper()

        if not robot.no_fuel():
            # getattr(k, self.action)()
            if self.action == "pick_up_hook":
                assert False
            elif self.action == "align":
                raw_obs = k.env.obs
                _, block_position, _, _ = k.get_obs()
                hook_target = np.array(
                    [block_position[0] - 0.5, block_position[1] - 0.05, 0.45]
                )
                action = get_move_action(raw_obs, hook_target, close_gripper=True)
                # pdb.set_trace()
                # k.render()
                if DSL_DEBUG:
                    print("[ACTION] align")
                _, rwd, done, info = k.step(action)
                # k.render()
            elif self.action == "sweep":
                raw_obs = k.env.obs
                _, block_position, _, place_position = k.get_obs()
                direction = np.subtract(place_position, block_position)
                direction = direction[:2] / np.linalg.norm(direction[:2])
                # print("[DIRECTION]", direction)
                action = np.array(
                    # [2.0 * direction[0], 2.0 * direction[1], 2.0 * direction[2], -1.0]
                    [3.0 * direction[0], 3.0 * direction[1], 0.0, -1.0]
                )
                if DSL_DEBUG:
                    print("[ACTION] sweep")
                _, rwd, done, info = k.step(action)
            elif self.action == "idle":
                rwd = 0
                k._elapsed_steps += 1
                if DSL_DEBUG:
                    print("[ACTION] idle action")
            # actions below are only used by library functions
            elif self.action == "move_to_object":
                raw_obs = k.env.obs
                # gripper_position, block_position, hook_position, place_position = k.get_obs()
                _, _, hook_position, _ = k.get_obs()
                target_position = get_target_position(hook_position)
                action = get_move_action(raw_obs, target_position)
                if DSL_DEBUG:
                    print("[ACTION] move to object")
                _, rwd, done, info = k.step(action)
            elif self.action == "open_gripper":
                open_gripper_action = np.array([0.0, 0.0, 0, 1.0])
                if DSL_DEBUG:
                    print("[ACTION] open gripper")
                _, rwd, done, info = k.step(open_gripper_action)
            elif self.action == "close_gripper":
                close_gripper_action = np.array([0.0, 0.0, 0, -1.0])
                if DSL_DEBUG:
                    print("[ACTION] close gripper")
                _, rwd, done, info = k.step(close_gripper_action)
            elif self.action == "move_down":
                raw_obs = k.env.obs
                _, _, hook_position, _ = k.get_obs()
                relative_grasp_position = (0.0, 0.0, -0.05)
                target_position = np.add(hook_position, relative_grasp_position)
                action = get_move_action(raw_obs, target_position)
                if DSL_DEBUG:
                    print("[ACTION] move down")
                _, rwd, done, info = k.step(action)
            elif self.action == "move_to_target":
                raw_obs = k.env.obs
                # gripper_position, block_position, hook_position, place_position = k.get_obs()
                _, _, hook_position, _ = k.get_obs()
                target_position = hook_position.copy()
                target_position[2] = 0.5
                action = get_move_action(raw_obs, target_position, close_gripper=True)
                if DSL_DEBUG:
                    print("[ACTION] move to target")
                _, rwd, done, info = k.step(action)
            if self.action != "idle":
                if DSL_DEBUG:
                    # pass
                    # data = k.render(mode="rgb_array")
                    k.render(mode="human")
                global g_counter
                global b_count
                global last_reward
                global last_success
                last_reward = rwd
                success = info["is_success"]
                last_success = success
                if b_count:
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
                print("hook_grasped", k.hook_grasped())
                print("hook_aligned", k.hook_aligned())
                print("object_at_target", k.object_at_target())
                print("object_below_gripper", k.object_below_gripper())
                print("gripper_are_open", k.gripper_are_open())
                print("object_inside_gripper", k.object_inside_gripper())
                print("object_is_grasped", k.object_is_grasped())
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
