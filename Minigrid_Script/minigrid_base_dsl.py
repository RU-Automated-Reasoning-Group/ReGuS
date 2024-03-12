import copy
import pdb
import time

from minigrid_implement.dsl import k_action, k_cond, k_cond_without_not

ENV = None

ACTION_DICT = None
ACTION_NAME = None
ACTION_LIST = None
WRAPPED_ACTION_LIST = None

COND_DICT = None
COND_NAME = None
COND_LIST = None
POSITIVE_COND_LIST = None

LIBRARY = []
LIBRARY_DICT = []


def set_library(library, library_dict):
    global LIBRARY, LIBRARY_DICT
    LIBRARY = library
    LIBRARY_DICT = library_dict


def set_env(env):
    global ENV
    ENV = env


def add_action(action_dict, action_list):
    for action in action_list:
        action_dict[action] = k_action(action)


def set_action_dict(env: str):
    action_dict = {}
    add_action(action_dict, ["move", "turn_left", "turn_right", "return"])

    if (
        env == "MiniGrid-RandomLavaCrossingS11N5-v0"
        or env == "MiniGrid-RandomCrossingS11N5-v0"
    ):
        pass
    elif env == "MiniGrid-MultiRoomNoDoor-N6-v0":
        add_action(action_dict, ["RC_get"])
    elif env == "MiniGrid-MultiRoom-N6-v0":
        add_action(action_dict, ["RC_get", "toggle"])
    elif env == "MiniGrid-LockedRoom-v0":
        add_action(
            action_dict, ["RC_get", "toggle", "get_key", "get_locked_door", "pickup"]
        )
        # action_dict.pop("turn_right")
    elif env.startswith("MiniGrid-DoorKey"):
        add_action(
            action_dict,
            ["RC_get", "toggle", "get_key", "get_locked_door", "pickup", "LK_get", "turn_right"],
        )
    elif env == "MiniGrid-PutNearTwoRoom-v0":
        add_action(
            action_dict,
            ["RC_get", "toggle", "get_key", "get_locked_door", "pickup", "LK_get", "turn_right", "DK_get", "drop", "get_ball"],
        )
    elif env == "MiniGrid-UnlockPickup-v0":
        add_action(
            action_dict,
            ["RC_get", "toggle", "get_key", "get_locked_door", "pickup", "LK_get", "turn_right", "DK_get", "drop", "get_ball", "put_near"],
        )
    else:
        assert False

    global ACTION_DICT, ACTION_NAME, ACTION_LIST, WRAPPED_ACTION_LIST
    ACTION_DICT = action_dict
    ACTION_NAME = [e for e in ACTION_DICT]
    ACTION_LIST = [ACTION_DICT[e] for e in ACTION_DICT]
    WRAPPED_ACTION_LIST = [ACTION(e) for e in ACTION_LIST]


def add_cond(cond_dict, cond_list):
    for a_cond in cond_list:
        cond_dict[a_cond] = k_cond(negation=False, cond=k_cond_without_not(a_cond))

        neg_a_cond = "not(" + a_cond + ")"
        cond_dict[neg_a_cond] = k_cond(negation=True, cond=k_cond_without_not(a_cond))


def set_cond_dict(env: str):
    cond_dict = {}
    add_cond(
        cond_dict,
        [
            "goal_present",
            "goal_on_left",
            "goal_on_right",
            "front_is_clear",
            "left_is_clear",
            "right_is_clear",
        ],
    )

    if env == "MiniGrid-RandomCrossingS11N5-v0":
        pass
    elif env == "MiniGrid-RandomLavaCrossingS11N5-v0":
        add_cond(cond_dict, ["front_is_lava"])
    elif env == "MiniGrid-MultiRoomNoDoor-N6-v0":
        add_cond(cond_dict, ["front_is_lava"])
    elif env == "MiniGrid-MultiRoom-N6-v0":
        add_cond(cond_dict, ["front_is_lava", "front_is_closed_door"])
    elif env == "MiniGrid-LockedRoom-v0":
        add_cond(
            cond_dict,
            [
                "front_is_lava",
                "front_is_closed_door",
                "front_is_locked_door",
                "front_is_key",
                "has_key",
            ],
        )
    elif env.startswith("MiniGrid-DoorKey"):
        add_cond(
            cond_dict,
            [
                "front_is_lava",
                "front_is_closed_door",
                "front_is_locked_door",
                "has_key",
                "front_is_key",
            ],
        )
    elif env == "MiniGrid-UnlockPickup-v0" or env == "MiniGrid-PutNearTwoRoom-v0":
        add_cond(
            cond_dict,
            [
                "front_is_ball",
                "front_is_lava",
                "front_is_closed_door",
                "front_is_locked_door",
                "has_key",
                "front_is_key",
                # "clear_to_drop",
            ],
        )      
    else:
        print(env)
        assert False

    global COND_DICT, COND_NAME, COND_LIST, POSITIVE_COND_LIST
    COND_DICT = cond_dict
    COND_NAME = [e for e in COND_DICT]
    COND_LIST = [COND_DICT[e] for e in COND_DICT]
    POSITIVE_COND_LIST = []
    for c, cond in COND_DICT.items():
        if not c.startswith("not"):
            POSITIVE_COND_LIST.append(cond)


# def set_abs_state(env):
#     class ABS_STATE:
#         def __init__(self):
#             self.state = {
#                 "front_is_clear": None,
#                 "left_is_clear": None,
#                 "right_is_clear": None,
#                 "goal_on_left": None,
#                 "goal_on_right": None,
#                 "goal_present": None,
#             }

#         def update(self, cond, description: str):
#             # description: T / F / DNC
#             self.state[str(cond)] = description

#         def __str__(self) -> str:
#             return str(self.state)

#     return ABS_STATE


def add_state(state_dict, state_list):
    for state in state_list:
        state_dict[state] = None


class ABS_STATE:
    def __init__(self):
        self.state = {}
        add_state(
            self.state,
            [
                "front_is_clear",
                "left_is_clear",
                "right_is_clear",
                "goal_on_left",
                "goal_on_right",
                "goal_present",
            ],
        )

        if ENV == "MiniGrid-RandomCrossingS11N5-v0":
            pass
        elif ENV == "MiniGrid-RandomLavaCrossingS11N5-v0":
            add_state(self.state, ["front_is_lava"])
        elif ENV == "MiniGrid-MultiRoomNoDoor-N6-v0":
            add_state(self.state, ["front_is_lava"])
        elif ENV == "MiniGrid-MultiRoom-N6-v0":
            add_state(self.state, ["front_is_lava", "front_is_closed_door"])
        elif ENV == "MiniGrid-LockedRoom-v0":
            add_state(
                self.state,
                [
                    "front_is_lava",
                    "front_is_closed_door",
                    "front_is_locked_door",
                    "front_is_key",
                    # "has_key",
                ],
            )
        elif ENV.startswith("MiniGrid-DoorKey"):
            add_state(
                self.state,
                [
                    "front_is_lava",
                    "front_is_closed_door",
                    "front_is_locked_door",
                    "has_key",
                    "front_is_key",
                ],
            )
        elif ENV == "MiniGrid-UnlockPickup-v0" or ENV == "MiniGrid-PutNearTwoRoom-v0":
            add_state(
                self.state,
                [
                    "front_is_lava",
                    "front_is_closed_door",
                    "front_is_locked_door",
                    "has_key",
                    "front_is_key",
                    "clear_to_drop",
                    "front_is_ball",
                ],
            )
        else:
            assert False

    def update(self, cond, description: str):
        # description: T / F / DNC
        self.state[str(cond)] = description

    def __str__(self) -> str:
        return str(self.state)

    def copy_over(self, old_abs_state):
        if old_abs_state is None:
            pdb.set_trace()
        for predicate in self.state:
            if predicate in old_abs_state.state:
                self.state[predicate] = old_abs_state.state[predicate]
            else:
                # set false for the new predicate
                # if predicate == "has_key":
                #     self.state[predicate] = 'DNC'
                # else:
                self.state[predicate] = "F"


def get_abs_state(robot):
    abs_state = ABS_STATE()
    for cond in POSITIVE_COND_LIST:
        if robot.execute_single_cond(cond):
            abs_state.update(cond, "T")
        else:
            abs_state.update(cond, "F")

    return abs_state


def satisfy_abs_state(current, required):
    satisfied = True
    for e in required.state:
        if required.state[e] == "DNC":  # does not care
            pass
        elif current.state[e] != required.state[e]:
            satisfied = False
            break

    return satisfied


def get_diff_abs_state(code_abs_state, obs_abs_state):
    diff_abs_state = []
    for e in code_abs_state.state:
        if obs_abs_state.state[e] != code_abs_state.state[e]:
            if code_abs_state.state[e] == "DNC":
                pass
            else:
                diff_abs_state.append(e)

    return diff_abs_state


def get_diff_conds(code_abs_state, obs_abs_state):
    diff_conds = []  # can be multiple conds
    for e in code_abs_state.state:
        if obs_abs_state.state[e] != code_abs_state.state[e]:
            if code_abs_state.state[e] == "DNC":
                pass
            elif code_abs_state.state[e] == "T":
                diff_conds.append(COND_DICT["not(" + e + ")"])
            elif code_abs_state.state[e] == "F":
                diff_conds.append(COND_DICT[e])

    return diff_conds


def get_neg_cond(cond):
    cond_name = str(cond)
    # print(cond_name)
    if "not" in cond_name:
        neg_cond_name = cond_name.replace("not", "").replace("(", "").replace(")", "")
        neg_cond_name = neg_cond_name.replace(" ", "")
        return COND_DICT[neg_cond_name]
    else:
        cond_name = cond_name.replace(" ", "")
        for key in COND_DICT:
            if "not" in key and cond_name in key:
                return COND_DICT[key]


def merge_abs_state(abs_state, new_abs_state):
    s = copy.deepcopy(abs_state)
    for e in s.state:
        if s.state[e] != new_abs_state.state[e]:
            s.state[e] = "DNC"

    return s


class ACTION:
    def __init__(self, action):
        self.abs_state = None
        self.action = action

        # NOTE: used for adding new IF branch
        self.break_point = False
        self.obs_abs_state = None
        self.post_abs_state = None
        self.resume_point = False
        self.returned = False

    def parameterize(self, from_object, to_object):
        pass

    def copy_over_abs_state(self):
        old_abs_state = self.abs_state
        old_obs_abs_state = self.obs_abs_state
        old_post_abs_state = self.post_abs_state

        if old_abs_state is None:
            self.abs_state = None
        else:
            self.abs_state = ABS_STATE()
            self.abs_state.copy_over(old_abs_state)

        if old_obs_abs_state is None:
            self.obs_abs_state = None
        else:
            self.obs_abs_state = ABS_STATE()
            self.obs_abs_state.copy_over(old_obs_abs_state)

        if old_post_abs_state is None:
            self.post_abs_state = None
        else:
            self.post_abs_state = ABS_STATE()
            self.post_abs_state.copy_over(old_post_abs_state)

    def remove_abs_state(self):
        self.abs_state = None
        self.obs_abs_state = None
        self.post_abs_state = None

    def execute(self, robot, stop):
        if robot.active:
            if str(self.action.action) in [
                "RC_get",
                "get_key",
                "get_locked_door",
                "LK_get",
                "DK_get",
                "get_ball",
                "get",
                "put_near",
            ]:
                if robot.force_execution:
                    # assert not self.resume_point
                    program = LIBRARY_DICT[self.action.action]
                    program.execute(robot)
                    robot.active = True
                    robot.returned = False
                elif not self.resume_point:
                    # init abstract state
                    if self.abs_state is None:
                        self.abs_state = get_abs_state(robot)

                    # check satisfy
                    if satisfy_abs_state(get_abs_state(robot), self.abs_state):
                        program = LIBRARY_DICT[self.action.action]
                        robot.force_execution = True
                        program.execute(robot)
                        robot.force_execution = False
                        robot.active = True
                        robot.returned = False

                        # modify post abstract state here
                        new_robot_state = get_abs_state(robot)

                        # init post abstract state
                        if self.post_abs_state is None:
                            self.post_abs_state = new_robot_state
                        # update post abstract state
                        elif not satisfy_abs_state(
                            new_robot_state, self.post_abs_state
                        ):
                            self.post_abs_state = merge_abs_state(
                                self.post_abs_state, new_robot_state
                            )
                    else:
                        self.break_point = True
                        self.bp_time = time.time()
                        self.obs_abs_state = get_abs_state(robot)
                        robot.active = False
                return
            if str(self.action.action) == "return":
                if self.resume_point:
                    return
                robot.active = False
                robot.returned = True
                return

            if robot.force_execution:  # without considering abs_state
                # assert not self.resume_point
                r = robot.execute_single_action(self.action)
                if r == -1:
                    # pdb.set_tace()
                    robot.active = False

                # robot.draw(log_print=True)
                # log_and_print("")
                # pdb.set_trace()

            elif not self.resume_point:
                # init abstract state
                if self.abs_state is None:
                    self.abs_state = get_abs_state(robot)

                # check satisfy
                if satisfy_abs_state(get_abs_state(robot), self.abs_state):
                    r = robot.execute_single_action(self.action)
                    # modify post abstrate state here (TODO: whether good to put here?)
                    new_robot_state = get_abs_state(robot)

                    # init post abstract state
                    if self.post_abs_state is None:
                        self.post_abs_state = new_robot_state
                    # update post abstract state
                    elif not satisfy_abs_state(new_robot_state, self.post_abs_state):
                        self.post_abs_state = merge_abs_state(
                            self.post_abs_state, new_robot_state
                        )

                    # NOTE: terminate when success (or failed)
                    if r == -1:
                        # pdb.set_tace()
                        robot.active = False

                # add break point
                else:
                    self.break_point = True
                    self.bp_time = time.time()
                    self.obs_abs_state = get_abs_state(robot)
                    robot.active = False

    def execute_and_update(self, robot):
        if robot.active:
            if self.abs_state is None:
                self.abs_state = get_abs_state(robot)
            new_abs_state = get_abs_state(robot)
            self.abs_state = merge_abs_state(self.abs_state, new_abs_state)

            if str(self.action.action) in LIBRARY_DICT:
                tmp_program = copy.deepcopy(LIBRARY_DICT[self.action.action])
                tmp_program.execute_and_update(robot)
                robot.active = True
                robot.returned = False
            elif str(self.action.action) != "return":
                r = robot.execute_single_action(self.action)
            else:
                robot.active = False

            if self.post_abs_state is None:
                self.post_abs_state = get_abs_state(robot)
            new_post_state = get_abs_state(robot)
            self.post_abs_state = merge_abs_state(self.post_abs_state, new_post_state)

    def reset_resume(self):
        self.resume_point = False

    def __str__(self):
        return str(self.action)

    def to_string_verbose(self):
        pre_str = "None" if self.abs_state is None else str(self.abs_state.state)
        post_str = (
            "None" if self.post_abs_state is None else str(self.post_abs_state.state)
        )
        return f"\n[\n{pre_str}\n {str(self.action)} \n{post_str}\n]"

    def pretty_print(self):
        pass


# store action used to end if branch
class HIDE_ACTION:
    def __init__(self, action):
        self.abs_state = None
        self.action = action

        self.break_point = False
        self.obs_abs_state = None
        self.post_abs_state = None

    def parameterize(self, from_object, to_object):
        pass

    def copy_over_abs_state(self):
        pass

    def remove_abs_state(self):
        self.abs_state = None
        self.obs_abs_state = None
        self.post_abs_state = None

    def execute_and_update(self, robot):
        pass

    def execute(self, robot, stop):
        pass

    def __str__(self):
        return ""

    def to_string_verbose(self):
        pre_str = "None" if self.abs_state is None else str(self.abs_state.state)
        post_str = (
            "None" if self.post_abs_state is None else str(self.post_abs_state.state)
        )
        return f"[{pre_str} hide_action ({self.action.to_string_verbose()}) {post_str}]"

    def pretty_print(self):
        pass


# search DSL
# S -> while B do S; S | C
# B -> conds

# NOTE: treate C as terminals
#       C does not contribute to the sketch


class C:
    def __init__(self, min_action=1, max_action="infty"):
        self.stmts = []
        self.touch = False
        self.min_action = min_action
        self.max_action = max_action

        self.resume_point = False

    def parameterize(self, from_object, to_object):
        for s in self.stmts:
            s.parameterize(from_object, to_object)

    def copy_over_abs_state(self):
        for s in self.stmts:
            s.copy_over_abs_state()

    def remove_abs_state(self):
        for s in self.stmts:
            s.remove_abs_state()

    # def execute(self, robot):
    #    raise NotImplementedError('Invalid code')
    def execute(self, robot, stop):
        if not self.resume_point:
            assert robot.active
            robot.active = False
            self.touch = True

    def execute_and_update(self, robot):
        if robot.active:
            for s in self.stmts:
                s.execute_and_update(robot)

    def reset_resume(self):
        self.resume_point = False

    def __str__(self):
        return f" C[{self.min_action}, {self.max_action}] "

    def to_string_verbose(self):
        return self.__str__()


class B:
    def __init__(self):
        self.cond = None

    def execute(self, robot):
        raise NotImplementedError("Invalid code")

    def __str__(self):
        return " B " if self.cond is None else str(self.cond)


class S:
    def __init__(self):
        self.stmts = []

    def parameterize(self, from_object, to_object):
        for s in self.stmts:
            s.parameterize(from_object, to_object)

    def copy_over_abs_state(self):
        for s in self.stmts:
            s.copy_over_abs_state()

    def remove_abs_state(self):
        for s in self.stmts:
            s.remove_abs_state()

    def execute(self, robot):
        if robot.active and not robot.no_fuel():
            # print('[execute S, i.e., execute nothing]')
            pass

    def execute_and_update(self, robot):
        pass

    def __str__(self):
        return " S "


class SN:
    def __init__(self):
        self.stmts = []

    def parameterize(self, from_object, to_object):
        pass

    def copy_over_abs_state(self):
        pass

    def remove_abs_state(self):
        pass

    def execute(self, robot):
        # raise NotImplementedError('Invalid code')
        if robot.active and not robot.no_fuel():
            # print('[execute S, i.e., execute nothing]')
            pass

    def execute_and_update(self, robot):
        pass

    def __str__(self):
        return "SN"


# NOTE: used to check if program complete
class END:
    def __init__(self):
        self.visited = False

    def parameterize(self, from_object, to_object):
        pass

    def copy_over_abs_state(self):
        pass

    def remove_abs_state(self):
        pass

    def execute(self, robot, stop):
        if robot.active:
            self.visited = True
            robot.active = False
            robot.returned = True

    def execute_and_update(self, robot):
        pass

    def __str__(self):
        return f"; END ({self.visited})"

    def to_string_verbose(self):
        return self.__str__()


class WHILE:
    def __init__(self):
        self.cond = [B()]
        self.stmts = [S()]
        self.robot_move = False
        self.resume_point = False
        self.resume_last = False
        self.start_pos = None

    def parameterize(self, from_object, to_object):
        self.cond[0].parameterize(from_object, to_object)
        for s in self.stmts:
            s.parameterize(from_object, to_object)

    def copy_over_abs_state(self):
        for s in self.stmts:
            s.copy_over_abs_state()

    def remove_abs_state(self):
        for s in self.stmts:
            s.remove_abs_state()

    def execute_with_plot(self, robot, stop):
        if robot.active and not robot.no_fuel():
            # check robot position
            if not self.resume_point:
                self.start_pos = tuple(robot.env.agent_pos)
            # do while
            while robot.active and (
                self.resume_point
                or (not robot.no_fuel() and robot.execute_single_cond(self.cond[0]))
            ):
                for s in self.stmts:
                    # NOTE: summarized as
                    if stop:
                        robot.draw()
                        pdb.set_trace()
                    s.execute(robot, stop)
                    # robot.env.render()
                    print(robot.steps)
                    if robot.steps == 238:
                        print("enter here")
                    # resume point
                    if self.resume_point:
                        if hasattr(s, "resume_point") and not s.resume_point:
                            self.reset_resume()

                    if not robot.active:
                        break

                # resume point
                if self.resume_last:
                    self.reset_resume()
                elif self.resume_point:
                    # pdb.set_trace()
                    break

                # debug test
                # robot.steps += 1
                # robot.draw()
                # pdb.set_trace()
            # check robot position
            end_pos = tuple(robot.env.agent_pos)
            # update
            if not self.resume_point:
                if self.start_pos != end_pos:
                    self.robot_move = True
                else:
                    self.robot_move = False

    def execute(self, robot, stop):
        if robot.active and not robot.no_fuel():
            # check robot position
            if not self.resume_point:
                self.start_pos = tuple(robot.env.agent_pos)
            # do while
            while robot.active and (
                self.resume_point
                or (not robot.no_fuel() and robot.execute_single_cond(self.cond[0]))
            ):
                for s in self.stmts:
                    # NOTE: summarized as
                    if stop:
                        robot.draw()
                        pdb.set_trace()
                    s.execute(robot, stop)

                    # resume point
                    if self.resume_point:
                        if hasattr(s, "resume_point") and not s.resume_point:
                            self.reset_resume()

                    if not robot.active:
                        break

                # resume point
                if self.resume_last:
                    self.reset_resume()
                elif self.resume_point:
                    # pdb.set_trace()
                    break

                # debug test
                robot.steps += 1
                # robot.draw()
                # pdb.set_trace()
            # check robot position
            end_pos = tuple(robot.env.agent_pos)
            # update
            if not self.resume_point:
                if self.start_pos != end_pos:
                    self.robot_move = True
                else:
                    self.robot_move = False

    def execute_and_update(self, robot):
        while robot.active and robot.execute_single_cond(self.cond[0]):
            for s in self.stmts:
                s.execute_and_update(robot)

    def reset_resume(self):
        # if str(self) == ' WHILE(not (markers_present)) { WHILE(not (markers_present)) { IF(left_is_clear) { turn_right}  move} ;} ;':
        #     pdb.set_trace()
        # elif str(self) == ' WHILE(not (markers_present)) { IF(left_is_clear) { turn_right}  move} ;':
        #     pdb.set_trace()
        self.resume_point = False
        self.resume_last = False
        for s in self.stmts:
            if hasattr(s, "resume_point"):
                if s.resume_point:
                    s.reset_resume()
                else:
                    break

    def __str__(self):
        string = ""
        string += " WHILE(" + str(self.cond[0]) + ") {"
        for s in self.stmts:
            string += str(s)
        string += "} ;"

        return string

    def to_string_verbose(self):
        string = ""
        string += " WHILE(" + str(self.cond[0]) + ") {"
        for s in self.stmts:
            string += s.to_string_verbose()
        string += "} ;"

        return string


# NOTE: we will not synthesize IF directly
class IF:
    def __init__(self, cond=None):
        self.cond = [B() if cond is None else cond]
        self.stmts = [C()]
        self.resume_point = False

    def parameterize(self, from_object, to_object):
        self.cond[0].parameterize(from_object, to_object)
        for s in self.stmts:
            s.parameterize(from_object, to_object)

    def copy_over_abs_state(self):
        for s in self.stmts:
            s.copy_over_abs_state()

    def remove_abs_state(self):
        for s in self.stmts:
            s.remove_abs_state()

    def execute(self, robot, stop):
        if robot.active and not robot.no_fuel():
            # IF
            if self.resume_point or robot.execute_single_cond(self.cond[0]):
                for s in self.stmts:
                    if stop:
                        robot.draw()
                        pdb.set_trace()
                    # NOTE: summarized as
                    s.execute(robot, stop)

                    # resume point
                    if self.resume_point:
                        if hasattr(s, "resume_point") and not s.resume_point:
                            self.reset_resume()

                    if not robot.active:
                        break

    def execute_and_update(self, robot):
        if robot.active:
            if robot.execute_single_cond(self.cond[0]):
                for s in self.stmts:
                    s.execute_and_update(robot)

    def reset_resume(self):
        self.resume_point = False
        for s in self.stmts:
            if hasattr(s, "resume_point"):
                if s.resume_point:
                    s.reset_resume()
                else:
                    # pdb.set_trace()
                    break

    def __str__(self):
        string = ""
        string += " IF(" + str(self.cond[0]) + ") {"
        for s in self.stmts:
            string += str(s)
        string += "} "

        return string

    def to_string_verbose(self):
        string = ""
        string += " IF(" + str(self.cond[0]) + ") {"
        for s in self.stmts:
            string += s.to_string_verbose()
        string += "} "

        return string


# NOTE: we will not synthesize IF directly
class IFELSE:
    def __init__(self, cond=None):
        self.cond = [B() if cond is None else cond]
        self.stmts = [C()]
        self.else_stmts = [C()]
        self.resume_point = False

    def parameterize(self, from_object, to_object):
        self.cond[0].parameterize(from_object, to_object)
        for s in self.stmts:
            s.parameterize(from_object, to_object)

        for s in self.else_stmts:
            s.parameterize(from_object, to_object)

    def copy_over_abs_state(self):
        for s in self.stmts:
            s.copy_over_abs_state()

        for s in self.else_stmts:
            s.copy_over_abs_state()

    def remove_abs_state(self):
        for s in self.stmts:
            s.remove_abs_tate()

        for s in self.else_stmts:
            s.remove_abs_state()

    def execute(self, robot, stop):
        if robot.active and not robot.no_fuel():
            into_if = False
            # IF
            if self.resume_point or robot.execute_single_cond(self.cond[0]):
                if not self.resume_point:
                    into_if = True
                for s in self.stmts:
                    if stop:
                        robot.draw()
                        pdb.set_trace()
                    # NOTE: summarized as
                    s.execute(robot, stop)

                    # resume point
                    if self.resume_point:
                        if hasattr(s, "resume_point") and not s.resume_point:
                            into_if = True
                            self.reset_resume()

                    if not robot.active:
                        break
            # ELSE
            if self.resume_point:
                assert not into_if
            if not into_if:
                for s in self.else_stmts:
                    if stop:
                        robot.draw()
                        pdb.set_trace()
                    # NOTE: summarized as
                    s.execute(robot, stop)

                    # resume point
                    if self.resume_point:
                        if hasattr(s, "resume_point") and not s.resume_point:
                            self.reset_resume()

                    if not robot.active:
                        break

    def execute_and_update(self, robot):
        if robot.active:
            if robot.execute_single_cond(self.cond[0]):
                for s in self.stmts:
                    s.execute_and_update(robot)
            else:
                for s in self.else_stmts:
                    s.execute_and_update(robot)

    def reset_resume(self):
        self.resume_point = False
        for s in self.stmts:
            if hasattr(s, "resume_point"):
                if s.resume_point:
                    s.reset_resume()
                else:
                    break
        for s in self.else_stmts:
            if hasattr(s, "resume_point"):
                if s.resume_point:
                    s.reset_resume()
                else:
                    break

    def __str__(self):
        string = ""
        string += " IF(" + str(self.cond[0]) + ") {"
        for s in self.stmts:
            string += str(s)
        string += "} "
        string += "ELSE {"
        for s in self.else_stmts:
            string += str(s)
        string += "}"

        return string

    def to_string_verbose(self):
        string = ""
        string += " IF(" + str(self.cond[0]) + ") {"
        for s in self.stmts:
            string += s.to_string_verbose()
        string += "} else { "
        for s in self.else_stmts:
            string += s.to_string_verbose()
        string += " }"

        return string


class Program:
    def __init__(self):
        self.stmts = [S(), END()]

    def parameterize(self, from_object, to_object):
        for s in self.stmts:
            s.parameterize(from_object, to_object)

    def copy_over_abs_state(self):
        for s in self.stmts:
            s.copy_over_abs_state()

    def remove_abs_state(self):
        for s in self.stmts:
            s.remove_abs_state()

    def execute(self, robot, stop=False):
        for s in self.stmts:
            if stop:
                robot.draw()
                pdb.set_trace()
            s.execute(robot, stop)
            # pdb.set_trace()
            if not robot.active:
                break

    def execute_and_update(self, robot):
        if robot.active:
            for s in self.stmts:
                s.execute_and_update(robot)

    def execute_with_plot(self, robot, stop=False):
        for s in self.stmts:
            if stop:
                robot.draw()
                pdb.set_trace()
            s.execute_with_plot(robot, stop)
            # pdb.set_trace()
            if not robot.active:
                break

    def complete(self):
        assert isinstance(self.stmts[-1], END)
        return self.stmts[-1].visited

    def reset(self):
        self.stmts[-1].visited = False

    def reset_c_touch(self):
        c_stmts, c_idx = self.find_actions(c_touch=True)
        while c_stmts is not None:
            c_stmts[c_idx].touch = False
            c_stmts, c_idx = self.find_actions(c_touch=True)

    # NOTE: find S / B
    def find(self):
        stmts, idx = self._find(self.stmts)
        if stmts is None:
            return None, None, None
        else:
            code = stmts[idx]
            if isinstance(code, S):
                code_type = "S"
            elif isinstance(code, B):
                code_type = "B"
            elif isinstance(code, SN):
                code_type = "SN"
            else:
                raise ValueError("Invalid code")
            return stmts, idx, code_type

    def _find(self, stmts):
        r_stmts, r_idx = None, None
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, SN)):
                return stmts, idx
            elif isinstance(code, (WHILE, IF, IFELSE)):
                r_stmts, r_idx = self._find(code.cond)
                if r_stmts is None:
                    r_stmts, r_idx = self._find(code.stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
            elif isinstance(code, (ACTION, k_cond, C, END, HIDE_ACTION)):
                pass
            else:
                import pdb
                pdb.set_trace()
                raise ValueError("Invalid code")

        return r_stmts, r_idx

    # NOTE: find cond that contains a C
    #       C presents in IF(cond) {C}, not just in WHILE
    def find_c_cond(self, c_touch=False):
        self.c_cond, self.c_cond_type = None, None
        self.found_c = False
        self.c_touch = c_touch
        self._find_c_cond(self.stmts)

        return self.c_cond, self.c_cond_type

    def _find_c_cond(self, stmts):
        for code in stmts:
            if isinstance(code, (WHILE, IF, IFELSE)):
                contains_c = False
                for s in code.stmts:
                    if isinstance(s, C):
                        if not self.c_touch or s.touch:
                            contains_c = True
                            break
                if not self.found_c and contains_c:
                    self.c_cond, self.c_cond_type = (
                        code.cond[0],
                        "w" if isinstance(code, WHILE) else "i",
                    )
                    self.found_c = True
                    return
                self._find_c_cond(code.stmts)
            elif isinstance(code, C):
                if not self.c_touch or code.touch:
                    self.found_c = True

    # find code containing C
    def find_c_stmt(self, cond_type, c_touch=False):
        self.found_c = False
        self.found_stmt = None
        self.c_touch = c_touch
        self._find_c_stmt(self.stmts, cond_type)
        assert self.found_stmt is not None

        return self.found_stmt

    def _find_c_stmt(self, stmts, cond_type):
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                pass
            elif isinstance(code, (IF, IFELSE)):
                self._find_c_stmt(code.stmts, cond_type)
                if self.found_c and self.found_stmt is None and cond_type == "i":
                    self.found_stmt = code
                if self.found_stmt is not None:
                    return
            elif isinstance(code, WHILE):
                self._find_c_stmt(code.stmts, cond_type)
                if self.found_c and self.found_stmt is None and cond_type == "w":
                    self.found_stmt = code
                if self.found_stmt is not None:
                    return
            elif isinstance(code, C):
                if not self.c_touch or code.touch:
                    self.found_c = True
                    return
            else:
                pdb.set_trace()
                raise ValueError("Invalide code")

    # NOTE: find C
    def find_actions(self, c_touch=False):
        self.c_touch = c_touch
        stmts, idx = self._find_actions(self.stmts)
        if stmts is None:
            return None, None
        else:
            code = stmts[idx]
            assert isinstance(code, C)
            return stmts, idx

    def _find_actions(self, stmts):
        r_stmts, r_idx = None, None
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                pass
            elif isinstance(code, (WHILE, IF)):
                r_stmts, r_idx = self._find_actions(code.stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
            elif isinstance(code, IFELSE):
                r_stmts, r_idx = self._find_actions(code.stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
                r_stmts, r_idx = self._find_actions(code.else_stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
            elif isinstance(code, C):
                if not self.c_touch or code.touch:
                    return stmts, idx
            else:
                pdb.set_trace()
                raise ValueError("Invalide code")

        return r_stmts, r_idx

    def find_break_point(self):
        stmts, idx = self._find_break_point(self.stmts)
        if stmts is None:
            return None, None
        else:
            code = stmts[idx]
            assert isinstance(code, ACTION) and code.break_point
            return stmts, idx

    def _find_break_point(self, stmts):
        r_stmts, r_idx = None, None
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, C, HIDE_ACTION, k_cond, END)):
                pass
            elif isinstance(code, (WHILE, IF)):
                r_stmts, r_idx = self._find_break_point(code.stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
            elif isinstance(code, IFELSE):
                r_stmts, r_idx = self._find_break_point(code.stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
                r_stmts, r_idx = self._find_break_point(code.else_stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
            elif isinstance(code, ACTION):
                if code.break_point:
                    return stmts, idx
            else:
                pdb.set_trace()
                raise ValueError("Invalide code")

        return r_stmts, r_idx

    # NOTE: expand sketch
    def expand(self):
        p_list = []
        new_prog = copy.deepcopy(self)
        stmts, idx, code_type = new_prog.find()
        # test parent sketch
        self_loop = False
        for code in new_prog.stmts:
            if isinstance(code, WHILE):
                self_loop = True
                break

        # expand B
        if code_type == "B":
            for cond in COND_LIST:
                stmts[idx] = copy.deepcopy(cond)
                p_list.append(copy.deepcopy(new_prog))

        # expand S
        elif code_type == "S":
            # S -> C
            if self_loop:
                stmts[idx] = C()
                p_list.append(copy.deepcopy(new_prog))
            # S -> while
            stmts[idx] = WHILE()
            # stmts.insert(idx + 1, SN())
            stmts.insert(idx + 1, S())
            if not self_loop:
                p_list.append(copy.deepcopy(new_prog))

            # add another C[1, 1]
            stmts.insert(idx, C(1, 1))
            p_list.append(copy.deepcopy(new_prog))

        # expand SN
        elif code_type == "SN":
            pdb.set_trace()
            # SN -> S
            stmts[idx] = S()
            p_list.append(copy.deepcopy(new_prog))
            # SN -> None
            stmts.pop(idx)
            p_list.append(copy.deepcopy(new_prog))

        else:
            pass

        return p_list

    # NOTE: expand C to actions
    def expand_actions(self, c_touch=False, while_drop=True, cond_type="w"):
        p_list = []
        action_list = []
        new_prog = copy.deepcopy(self)
        # debug (for now, c_touch should be True)
        assert c_touch

        stmts, idx = new_prog.find_actions(c_touch)
        if not stmts is None:
            current_C = stmts[idx]
            if cond_type == "i":
                # expand in if, add return as one of the actions
                lst = copy.deepcopy(WRAPPED_ACTION_LIST)
                lst.append(ACTION(k_action("return")))
            else:
                lst = copy.deepcopy(WRAPPED_ACTION_LIST)
            for action in lst:
                if current_C.max_action != "infty":
                    if current_C.max_action == 1:
                        stmts[idx] = copy.deepcopy(action)
                        p_list.append(copy.deepcopy(new_prog))
                        action_list.append(copy.deepcopy(action))
                    else:
                        stmts[idx] = copy.deepcopy(action)
                        stmts.insert(
                            idx + 1,
                            C(min_action=1, max_action=current_C.max_action - 1),
                        )
                        p_list.append(copy.deepcopy(new_prog))
                        action_list.append(copy.deepcopy(action))
                        stmts.pop(idx + 1)
                else:
                    stmts[idx] = copy.deepcopy(action)
                    if str(action.action.action) != "return":
                        stmts.insert(idx + 1, C())
                    p_list.append(copy.deepcopy(new_prog))
                    action_list.append(copy.deepcopy(action))
                    if str(action.action.action) != "return":
                        stmts.pop(idx + 1)
            # attempt to drop C when While;C
            # only drop the last C but not C between two while loops
            if (
                while_drop
                and idx > 0
                and isinstance(stmts[idx - 1], WHILE)
                and idx == len(stmts) - 1
            ):
                stmts.pop(idx)
                p_list.append(copy.deepcopy(new_prog))
                action_list.append(None)
                stmts.insert(idx, C())
        else:
            pass

        return p_list, action_list

    def __str__(self):
        string = ""
        for s in self.stmts:
            string += str(s)

        return string

    def to_string_verbose(self):
        string = ""
        for s in self.stmts:
            string += s.to_string_verbose()
        return string

    # count C amount
    def count_C(self):
        self.count = 0
        self._count_C(self.stmts)

        return self.count

    def _count_C(self, stmts):
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                pass
            elif isinstance(code, (WHILE, IF, IFELSE)):
                self._count_C(code.stmts)
            elif isinstance(code, C):
                self.count += 1
            else:
                pdb.set_trace()
                raise ValueError("Invalide code")

    # set resume point (c touch or break point)
    def set_resume_points(self):
        self.found = False
        path = self._set_resume_point(self.stmts, [])
        if not self.found:
            print("no c touch or break point")
            pdb.set_trace()
            print("solve?")
        else:
            for code in path:
                if hasattr(code, "resume_point"):
                    code.resume_point = True

    def _set_resume_point(self, stmts, path):
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                if code.break_point:
                    # print("why break point first??")
                    # pdb.set_trace()
                    # print("solve?")
                    self.found = True
                    return path

                path.append(code)
                continue

            elif isinstance(code, IF):
                path.append(code)
                path = self._set_resume_point(code.stmts, path)
                if self.found:
                    return path

            elif isinstance(code, IFELSE):
                path.append(code)
                path = self._set_resume_point(code.stmts, path)
                if self.found:
                    return path
                path = self._set_resume_point(code.else_stmts, path)
                if self.found:
                    return path

            elif isinstance(code, WHILE):
                path.append(code)
                path = self._set_resume_point(code.stmts, path)
                if self.found:
                    # special case
                    if isinstance(code.stmts[-1], C) and code.stmts[-1].touch:
                        code.resume_last = True
                    return path

            elif isinstance(code, C):
                if code.touch:
                    self.found = True
                    return path
                else:
                    path.append(code)

            else:
                pdb.set_trace()
                raise ValueError("Invalide code")

        return path

    # check resume point
    def check_resume_points(self):
        self.found_resume = False
        self._check_resume_points(self.stmts)

        return self.found_resume

    def _check_resume_points(self, stmts):
        for idx, code in enumerate(stmts):
            if hasattr(code, "resume_point") and code.resume_point:
                self.found_resume = True
                return

            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                continue

            elif isinstance(code, (IF, IFELSE)):
                self._check_resume_points(code.stmts)
                if self.found_resume:
                    return
                if isinstance(code, IFELSE):
                    self._check_resume_points(code.else_stmts)
                    if self.found_resume:
                        return

            elif isinstance(code, WHILE):
                self._check_resume_points(code.stmts)
                if self.found_resume:
                    return

            elif isinstance(code, C):
                continue

            else:
                pdb.set_trace()
                raise ValueError("Invalide code")

    # reset resume point
    def reset_resume_points(self):
        self.found_resume = False
        self._reset_resume_points(self.stmts)

        return self.found_resume

    def _reset_resume_points(self, stmts):
        for idx, code in enumerate(stmts):
            if hasattr(code, "resume_point") and code.resume_point:
                code.resume_point = False

            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                continue

            elif isinstance(code, (IF, IFELSE)):
                self._reset_resume_points(code.stmts)
                if isinstance(code, IFELSE):
                    self._reset_resume_points(code.else_stmts)

            elif isinstance(code, WHILE):
                self._reset_resume_points(code.stmts)

            elif isinstance(code, C):
                continue

            else:
                pdb.set_trace()
                raise ValueError("Invalide code")


def get_abs_state_from_list(lst):
    state = ABS_STATE()
    if lst is None:
        return state
    state.state["front_is_clear"] = lst[0]
    state.state["left_is_clear"] = lst[1]
    state.state["right_is_clear"] = lst[2]

    state.state["goal_on_left"] = lst[3]
    state.state["goal_on_right"] = lst[4]
    state.state["goal_present"] = lst[5]

    return state


def get_cond(negation, cond):
    return k_cond(negation=negation, cond=k_cond_without_not(cond))


def get_action(action, abs_state, post_abs_state):
    action = ACTION(k_action(action))
    action.abs_state = abs_state
    action.post_abs_state = post_abs_state
    return action
