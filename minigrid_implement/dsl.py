

DSL_DEBUG = False
g_counter = None
search_counter = None
SEARCH_STATUS = True

def print_interaction():
    global search_counter
    print(f"the number of interaction is {search_counter}")

class k_cond:
    """cond : cond_without_not
    | NOT C_LBRACE cond_without_not C_RBRACE
    """

    def __init__(self, negation: bool, cond):
        self.negation = negation
        self.cond = cond

    def parameterize(self, from_object, to_object):
        self.cond.parameterize(from_object, to_object)

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

    def parameterize(self, from_object, to_object):
        self.cond = self.cond.replace(from_object, to_object)

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

        if not robot.no_fuel():
            # getattr(k, self.action)()
            robot.steps += 1

            if self.action == "move":
                _, rwd, _, _, _ = k.step(k.actions.forward)
                if DSL_DEBUG:
                    print("move")
            elif self.action == "turn_left":
                _, rwd, _, _, _ = k.step(k.actions.left)
                if DSL_DEBUG:
                    print("turn_left")
            elif self.action == "turn_right":
                _, rwd, _, _, _ = k.step(k.actions.right)
                if DSL_DEBUG:
                    print("turn_right")
            elif self.action == "toggle":
                _, rwd, _, _, _ = k.step(k.actions.toggle)
                if DSL_DEBUG:
                    print("toggle")
            elif self.action == "pickup":
                _, rwd, _, _, _ = k.step(k.actions.pickup)
            elif self.action == "drop":
                _, rwd, _, _, _ = k.step(k.actions.drop)
            else:
                import pdb

                pdb.set_trace()
            if SEARCH_STATUS:
                # used to counter the number of environment interactions
                global search_counter
                search_counter = 0 if search_counter is None else search_counter + 1
                # print(f"number of interaction is {search_counter}")
            if DSL_DEBUG:
                global g_counter
                if g_counter is None:
                    g_counter = 0
                else:
                    g_counter += 1
                dir = f"frames/img{g_counter:06d}.png"
                k.env.env.render(dir=dir)
            # if robot:
            #     # robot.steps += 1
            #     # robot.action_steps += 1
            #     if (
            #         str(self.action) == "get_key"
            #         or str(self.action) == "get_door"
            #         or str(self.action) == "get_simple"
            #     ):
            #         # return 0
            #         import pdb

            #         pdb.set_trace()
            #     if robot.reward == 1 and self.action in [
            #         "pickup",
            #         "toggle",
            #         "turn_left",
            #         "turn_right",
            #     ]:
            #         # the reward is already 1 and the action will not move to other location
            #         assert rwd == 0 or rwd == -1
            #         return robot.reward
            #     robot.reward = rwd
            #     if rwd == 1:
            #         robot.reach_goal_times += 1
            #     return robot.reward
            # if robot.steps >= robot.max_steps:
            #    robot.no_fuel = True
            # reward = robot.check_reward()
            # if robot.is_accumulated:
            #     robot.acc(reward)
            # return reward
            return rwd
        else:
            return robot.reward

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
