from numpy.lib.histograms import histogram
import torch
import torch.nn as nn
import numpy as np
import copy


from .karel import Karel
from .generator import KarelStateGenerator
from .counter import KarelExecuteCounter
from .logger import *
from .dsl import *
from network.predictor import CodePredictor


# wrapper class for karel robot
class KarelRobotEnv:
    def __init__(self, env, seed=999):
        
        self.env = env
        self.seed = seed
        self.gen = KarelStateGenerator(seed=self.seed)

        self.init_state = self.state_init(self.gen, self.env)
        self.karel = Karel(state=self.init_state)

    def state_init(self, gen, env):

        gen_function_base = "generate_single_state_"

        if env == "cleanHouse":
            gen_function = gen_function_base + "clean_house"
        elif env == "harvester":
            gen_function = gen_function_base + "harvester"
        elif env == "randomMaze":
            gen_function = gen_function_base + "random_maze"
        elif env == "fourCorners":
            gen_function = gen_function_base + "four_corners"
        elif env == "stairClimber":
            gen_function = gen_function_base + "stair_climber"
        elif env == "topOff":
            gen_function = gen_function_base + "top_off"

        state, _, _, _, _ = getattr(gen, gen_function)()
        
        return state.astype(int)

    def get_state(self):
        return self.karel.state

    def execute(self, program, states_logger=None, max_steps=None):
        exe_counter = None if not max_steps else KarelExecuteCounter(max_steps)
        program(self.karel, states_logger, exe_counter)

        return states_logger

    def execute_single_stmt(self, stmt):
        tmp_program = k_prog(k_stmt(stmt))
        tmp_program(self.karel)

    def execute_single_cond(self, cond):
        return cond(self.karel)

    def draw(self, *args, **kwargs):
        return self.karel.draw(*args, **kwargs)

# BASE karel robot parallel executor
class KarelRobotEnvExecutor(nn.Module):
    def __init__(self, env, batch_size, seed):
        super().__init__()
        
        self.env = env
        self.batch_size = batch_size
        self.seed = seed
        
        self.robots = []
        self.exe_robots = []  # for final execution
        self.loggers = []
        
        if env == "cleanHouse":
            self.model = CodePredictor(h=14, w=22)
            logger = CleanHouseLogger
        elif env == "harvester":
            self.model = CodePredictor(h=8, w=8)
            logger = HarvesterLogger
        elif env == "randomMaze":
            self.model = CodePredictor(h=8, w=8)
            logger = RandomMazeLogger
        elif env == "fourCorners":
            self.model = CodePredictor(h=12, w=12)
            logger = FourCornerLogger
        elif env == "stairClimber":
            self.model = CodePredictor(h=12, w=12)
            logger = StairClimberLogger
        elif env == "topOff":
            self.model = CodePredictor(h=12, w=12)
            logger = TopOffLogger

        for i in range(self.batch_size):
            self.robots.append(KarelRobotEnv(env=self.env, seed=self.seed + i))
            self.exe_robots.append(KarelRobotEnv(env=self.env, seed=self.seed + i))
            self.loggers.append(logger())

        # init the program
        self.program = k_prog(
                            k_stmt(
                                #k_place_holder(self.robots_states)
                                k_place_holder(list(range(self.batch_size)))
                            )
                        )

    # current robot states
    def current_robot_state(self, index):
        return self.robots[index].get_state()

    # TODO: suport multiprocess in the future
    def execute(self, index=None, max_steps=None):
        exe_range = range(self.batch_size) if index is None else index
        for i in exe_range:
            # execute and save the state
            program = copy.deepcopy(self.program)
            self.exe_robots[i].execute(self.program, self.loggers[i], max_steps)

    # will not execute loops
    def execute_single_stmt(self, stmt, index=None):
        exe_range = range(self.batch_size) if index is None else index
        for i in exe_range:
            # execute the single stmt
            self.robots[i].execute_single_stmt(stmt)

    # execute a cond and return true and false index
    def execute_single_cond(self, cond, index=None):
        true_index, false_index = [], []
        exe_range = range(self.batch_size) if index is None else index
        for i in exe_range:
            if (self.robots[i].execute_single_cond(cond)):
                true_index.append(i)
            else:
                false_index.append(i)

        return true_index, false_index

    # locate the place holder and predict the stmt
    def predict(self, mask=None):

        # place holder hook
        ph_hook = self.program.register()
        if not ph_hook:
            # program generation complete
            code, mask, done = None, None, True
            return code, mask, done
        
        # place holder and states (numpy)
        ph = ph_hook.function
        ph_index = ph.index

        # stacked states (tensor)
        ph_states_tensors = []
        for i in ph_index:
            ph_states_tensors.append(torch.from_numpy(self.current_robot_state(i)))
        
        stacked_states = torch.stack(ph_states_tensors, dim=0)  
        stacked_states = stacked_states.permute(0, 3, 1, 2) # N x C x H x W

        code, prediction_index = self.model(stacked_states.float(), mask)
        mask = None
        done = False

        if not isinstance(code, k_stmt_stmt):
        
            if isinstance(code, k_while):
                print('[prediction] single while')
                true_index, false_index = self.execute_single_cond(cond=code.cond, index=ph_index)
                code.stmt.function.index = true_index
                if not true_index:
                    code = k_null()
                mask = prediction_index

            elif isinstance(code, k_if):
                print('[prediction] single if')
                true_index, false_index = self.execute_single_cond(cond=code.cond, index=ph_index)
                code.stmt.function.index = true_index
                if not true_index:
                    code = k_null()
                mask = prediction_index

            elif isinstance(code, k_ifelse):
                print('[prediction] single ifelse')
                true_index, false_index = self.execute_single_cond(cond=code.cond, index=ph_index)
                code.stmt1.function.index = true_index
                code.stmt2.function.index = false_index
                if not true_index:
                    code.stmt1 = k_null()
                if not false_index:
                    code.stmt2 = k_null()
                mask = prediction_index

            else:
                # execute the newly generated code
                print('[prediction] single others')
                self.execute_single_stmt(stmt=code, index=ph_index)

        else:
            
            if isinstance(code.stmt1, k_while):
                print('[prediction] ss while')
                true_index, false_index = self.execute_single_cond(cond=code.stmt1.cond, index=ph_index)
                code.stmt1.stmt.function.index = true_index
                code.stmt2.function.index = ph_index
                if not true_index:
                    code.stmt1 = k_null()
                mask = prediction_index

            elif isinstance(code.stmt1, k_if):
                print('[prediction] ss if')
                true_index, false_index = self.execute_single_cond(cond=code.stmt1.cond, index=ph_index)
                code.stmt1.stmt.function.index = true_index
                code.stmt2.function.index = ph_index
                if not true_index:
                    code.stmt1 = k_null()
                mask = prediction_index

            elif isinstance(code.stmt1, k_ifelse):
                print('[prediction] ss ifelse')
                true_index, false_index = self.execute_single_cond(cond=code.stmt1.cond, index=ph_index)
                code.stmt1.stmt1.function.index = true_index
                code.stmt1.stmt2.function.index = false_index
                code.stmt2.function.index = ph_index
                if not true_index:
                    code.stmt1.stmt1 = k_null()
                if not false_index:
                    code.stmt1.stmt2 = k_null()
                mask = prediction_index

            else:
                # execute the newly generated code
                # save the new ph_index for future prediction
                print('[prediction] ss others')
                self.execute_single_stmt(stmt=code.stmt1, index=ph_index)
                code.stmt2.function.index = ph_index

        return code, mask, done

    def mount(self, code):
        # place holder hook
        ph_hook = self.program.register()

        # mount the code
        ph_hook.function = copy.deepcopy(code)

    def compute_reward(self):
        raise NotImplementedError

    def get_histories(self):
        # all states histories from loggers
        histories = []
        for i in range(self.batch_size):
            history = self.loggers[i].get_history()
            histories.append(history)

        return histories


class KarelRobotRandomMaze(KarelRobotEnvExecutor):
    def __init__(self, batch_size, seed):
        super().__init__(env='randomMaze', batch_size=batch_size, seed=seed)

    def compute_reward(self):
        total_rewards = 0
        for i in range(self.batch_size):
            total_rewards += self.loggers[i].compute_reward()

        return total_rewards/self.batch_size


class KarelRobotCleanHouse(KarelRobotEnvExecutor):
    def __init__(self, batch_size, seed):
        super().__init__(env='cleanHouse', batch_size=batch_size, seed=seed)

    def compute_reward(self):
        total_rewards = 0
        for i in range(self.batch_size):
            logger = self.loggers[i]
            if logger.compute_reward() == 1:
                total_rewards += 1
            else:
                state = logger.get_history()[-1]
                num_cleaned = 0
                for pos in logger.markers_pos:
                    if not state[pos[0]][pos[1]][6] and state[pos[0]][pos[1]][5]:
                        num_cleaned += 1
                total_rewards += num_cleaned / len(logger.markers_pos)

        return total_rewards / self.batch_size


class KarelRobotHarvester(KarelRobotEnvExecutor):
    def __init__(self, batch_size, seed):
        super().__init__(env='harvester', batch_size=batch_size, seed=seed)

    def compute_reward(self):
        total_rewards = 0
        for i in range(self.batch_size):
            logger = self.loggers[i]
            if logger.compute_reward() == 1:
                total_rewards += 1
            else:
                state = logger.get_history()[-1]
                total_markers = logger.total_markers
                final_markers = np.sum(state[:, :, 6:])
                total_rewards += (total_markers - final_markers) / total_markers
        
        return total_rewards / self.batch_size


class KarelRobotFourCorners(KarelRobotEnvExecutor):
    def __init__(self, batch_size, seed):
        super().__init__(env='fourCorners', batch_size=batch_size, seed=seed)

    def compute_reward(self):
        
        total_rewards = 0
        for i in range(self.batch_size):
            if self.loggers[i].compute_reward == 1:
                total_rewards += 1
            else:
                state = self.loggers[i].get_history()[-1]
                reward = 0
                total_markers = np.sum(state[:,:,6:])

                if state[1, 1, 6]:
                    reward += 0.25
                if state[6, 1, 6]:
                    reward += 0.25
                if state[6, 6, 6]:
                    reward += 0.25
                if state[1, 6, 6]:
                    reward += 0.25

                correct_markers = int(reward * 4)
                incorrect_markers = total_markers - correct_markers
                if incorrect_markers > 0:
                    reward = 0
                
                total_rewards += reward

        return total_rewards / self.batch_size


class KarelRobotStairClimer(KarelRobotEnvExecutor):
    def __init__(self, batch_size, seed):
        super().__init__(env='stairClimber', batch_size=batch_size, seed=seed)

    def compute_reward(self):
        total_rewards = 0
        for i in range(self.batch_size):
            total_rewards += self.loggers[i].compute_reward()

        return total_rewards/self.batch_size


class KarelRobotTopOff(KarelRobotEnvExecutor):
    def __init__(self, batch_size, seed):
        super().__init__(env='topOff', batch_size=batch_size, seed=seed)

    def compute_reward(self):
        total_rewards = 0
        for i in range(self.batch_size):
            logger = self.loggers[i]
            if logger.compute_reward() == 1:
                total_rewards += 1
            else:
                reward = 0
                state = logger.get_history()[-1]
                for m in range(1, state.shape[1] - 1):
                    pos = [state.shape[0] - 2, m]
                    if pos in logger.markers_pos:
                        if state[pos[0]][pos[1]][7]:
                            reward += 1
                        else:
                            break
                    else:
                        if not state[pos[0]][pos[1]][5]:
                            reward += 1
                        else:
                            break
                
                if logger.hero_pos(state) == [state.shape[0] - 2, state.shape[1] - 2]:
                    reward += 1
                total_rewards += reward / (state.shape[1] - 1)

        return total_rewards/self.batch_size