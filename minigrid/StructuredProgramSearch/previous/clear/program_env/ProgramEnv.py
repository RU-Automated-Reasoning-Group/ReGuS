import torch
import torch.nn as nn

import gym
from gym import spaces
import numpy as np


from karel.robot import BatchedKarelRobots
from karel.dsl import *


ACTION_INDEX = [0, 1, 2, 3, 4]
ACTION_NAME = [
    'move',
    'turn_right',
    'turn_left',
    'pick_marker',
    'put_marker'
]

COND_INDEX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
COND_CODE = [
    # cond
    k_cond(negation=False, cond=k_cond_without_not('front_is_clear')),
    k_cond(negation=False, cond=k_cond_without_not('left_is_clear')),
    k_cond(negation=False, cond=k_cond_without_not('right_is_clear')),
    k_cond(negation=False, cond=k_cond_without_not('markers_present')),
    k_cond(negation=False, cond=k_cond_without_not('no_markers_present')),
    # NOT cond
    k_cond(negation=True, cond=k_cond_without_not('front_is_clear')),
    k_cond(negation=True, cond=k_cond_without_not('left_is_clear')),
    k_cond(negation=True, cond=k_cond_without_not('right_is_clear')),
    k_cond(negation=True, cond=k_cond_without_not('markers_present')),
    k_cond(negation=True, cond=k_cond_without_not('no_markers_present')),
]
COND_TRUE_INDEX = [5, 6, 7, 8, 9]
COND_FALSE_INDEX = [10, 11, 12, 13, 14]

IF_INDEX = [15]
IFELSE_INDEX = [16]
WHILE_INDEX = [17]
END_INDEX = [18]
SKETCH_INDEX = [15, 16, 17, 18]


class ProgramEnv(gym.Env):
    def __init__(self, task, seed, batch_size, print_program=True):
        
        self.action_space = spaces.Discrete(19)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float)

        self.task = task
        self.seed = seed
        self.batch_size = batch_size
        self.print_program = print_program
        
        if task == 'cleanHouse':
            self.encoder = EnvEncoder(h=14, w=22)
        elif task == 'harvester':
            self.encoder = EnvEncoder(h=8, w=8)
        elif task == 'randomMaze':
            self.encoder = EnvEncoder(h=8, w=8)
        elif task == 'fourCorners':
            self.encoder = EnvEncoder(h=12, w=12)
        elif task == 'stairClimber':
            self.encoder = EnvEncoder(h=12, w=12)
        elif task == 'topOff':
            self.encoder = EnvEncoder(h=12, w=12)

        self.robots = BatchedKarelRobots(task=self.task, seed=self.seed, batch_size=batch_size)

        self.program = Program(list(range(self.batch_size)))

        # under construction: add sketch
        while_code = k_while()
        while_code.cond = k_cond(negation=False, cond=k_cond_without_not('no_markers_present'))
        while_code.stmts = [k_place_holder(candidates=list(range(self.batch_size)), end_for_while=True, cond=while_code.cond)]
        #self.program.program.stmts = [while_code, k_place_holder(list(range(self.batch_size)))]
        self.program.program.stmts = [while_code, k_end()]        

    def step(self, action):

        candidates, handler = locate(self.program)

        # TODO: program is finished
        if not handler:
            observation = self._get_obs()
            mask = []
            reward = 0
            done = True
            info = {}
            if self.print_program:
                print(self.program.program)
            return observation, mask, reward, done, info

        # TODO: no fuel
        if self.robots.any_no_fuel():
            print('any no fuel')
            observation = self._get_obs()
            mask = []
            reward = 0
            done = True
            info = {}
            if self.print_program:
                print(self.program.program)
            return observation, mask, reward, done, info
        
        # get latent states observations
        observation = self._get_obs(candidates=candidates)
        
        # expand the program and possibly execute
        # mask if needed
        reward, mask = self.program.step(action, self.robots, candidates, handler)

        info = {}
        
        # TODO: how to terminate?
        #     : 1) 2 means all success
        #     : 2) -1 means all failed (currently)
        #     : 3) what if only some of instance failed ?
        #     : 4) == -1, terminate ; < 0, terminate ; or simply do not terminate
        done = True if reward == 2 else False

        #print('[reward]', reward, '[program]', self.program.program)
        return observation, mask, reward, done, info


    def _get_obs(self, candidates=None):

        return self.robots.get_latent_state(encoder=self.encoder, candidates=candidates).detach().numpy()

    def reset(self):
        self.seed += 1
        self.robots = BatchedKarelRobots(task=self.task, seed=self.seed, batch_size=self.batch_size)
        self.program = Program(list(range(self.batch_size)))

        # under construction: add sketch
        while_code = k_while()
        while_code.cond = k_cond(negation=False, cond=k_cond_without_not('no_markers_present'))
        while_code.stmts = [k_place_holder(candidates=list(range(self.batch_size)), end_for_while=True, cond=while_code.cond)]
        #self.program.program.stmts = [while_code, k_place_holder(list(range(self.batch_size)))]
        self.program.program.stmts = [while_code, k_end()]        

        
        return self._get_obs(list(range(self.batch_size)))


# find the next ph to be expanded
def _locate(stmts: list):
    for s in stmts:
        if isinstance(s, k_place_holder):
            return s.candidates, stmts
        
        elif isinstance(s, (k_action, k_end)):
            pass
        
        elif isinstance(s, (k_while, k_if)):
            if s.requires_cond():
                assert isinstance(s.cond, k_place_holder)
                return s.cond.candidates, s
            elif s.finished():
                pass
            else:
                return _locate(s.stmts)

        elif isinstance(s, k_ifelse):
            if s.requires_cond():
                assert isinstance(s.cond, k_place_holder)
                return s.cond.candidates, s
            elif s.finished():
                pass
            else:
                result = _locate(s.stmts1)
                if not result is None:
                    return result
                else:
                    return _locate(s.stmts2)

    # TODO: program complete
    return [], None
                

def locate(program):
    return _locate(program.program.stmts)


class Program:
    def __init__(self, candidates):
        self.program = k_prog(candidates)
        self.prev_sketch = []
        self.mask = COND_INDEX

    def step(self, action, robots, candidates, handler):
    
        reward = 0
        action = int(action.item())
        
        if action in ACTION_INDEX:
            
            code = k_action(ACTION_NAME[action])
            for i in range(len(handler)):
                if isinstance(handler[i], k_place_holder):
                    handler.insert(i, code)
                    break
            
            # execute
            reward = robots.execute_single_action(code, candidates)
            #print(reward)
            
            self.mask = COND_INDEX
            self.mask = COND_INDEX + [15, 16, 17] # only action + end are allowed
        
        elif action in COND_INDEX:
            
            code = COND_CODE[action - 5]
            
            assert isinstance(handler, (k_if, k_ifelse, k_while))
            handler.cond = code
            
            true_candidates, false_candidates = robots.execute_single_cond(code, candidates)
            
            if isinstance(handler, k_while):
                if true_candidates:
                    handler.stmts.append(k_place_holder(true_candidates, end_for_while=True, cond=handler.cond))
                else:
                    #handler.stmts.append(k_end())
                    handler.cond = k_cond(negation=True, cond=code)
                    handler.stmts.append(k_place_holder(false_candidates, end_for_while=True, cond=handler.cond))

            elif isinstance(handler, k_if):
                if true_candidates:
                    handler.stmts.append(k_place_holder(true_candidates))
                else:
                    #handler.stmts.append(k_end())
                    handler.cond = k_cond(negation=True, cond=code)
                    handler.stmts.append(k_place_holder(false_candidates, end_for_while=True, cond=handler.cond))

            elif isinstance(handler, k_ifelse):
                if true_candidates:
                    handler.stmts1.append(k_place_holder(true_candidates))
                else:
                    handler.stmts1.append(k_end())
                if false_candidates:
                    handler.stmts2.append(k_place_holder(false_candidates))
                else:
                    handler.stmts1.append(k_end())
            
            self.mask = COND_INDEX + self.prev_sketch + END_INDEX # 1) don't repeat 2) don't terminate directly


        elif action in IF_INDEX:

            code = k_if()
            code.cond = k_place_holder(candidates)
            for i in range(len(handler)):
                if isinstance(handler[i], k_place_holder):
                    handler.insert(i, code)
                    break

            self.mask = ACTION_INDEX + SKETCH_INDEX
            self.prev_sketch = IF_INDEX

        elif action in IFELSE_INDEX:

            code = k_ifelse()
            code.cond = k_place_holder(candidates)
            for i in range(len(handler)):
                if isinstance(handler[i], k_place_holder):
                    handler.insert(i, code)
                    break
            
            self.mask = ACTION_INDEX + SKETCH_INDEX
            self.prev_sketch = IFELSE_INDEX

        elif action in WHILE_INDEX:
            
            code = k_while()
            code.cond = k_place_holder(candidates)
            for i in range(len(handler)):
                if isinstance(handler[i], k_place_holder):
                    handler.insert(i, code)
                    break
            
            self.mask = ACTION_INDEX + SKETCH_INDEX
            self.prev_sketch = WHILE_INDEX
        
        # a lot of things to do here
        # TODO: restart
        # TODO: rewards in restart
        # TODO: success in restart
        # TODO: I think the problem is 
        #     : 1) no fuel during restart
        #     : 2) success during restart
        #     : 3) others, how to compute the reward?
        elif action in END_INDEX:

            end_for_while = False
            cond = None

            # First we insert
            for i in range(len(handler)):
                if isinstance(handler[i], k_place_holder):
                    code = k_end()
                    if handler[i].end_for_while:
                        end_for_while = True
                        cond = handler[i].cond
                    handler[i] = code

            # Then we restart while loop
            if end_for_while and len(handler) > 1:
                candidates_reward = 0
                for i in candidates:
                    robot = robots.robots[i]
                    robot.start_acc()
                    while robot.execute_single_cond(cond) and not robot.no_fuel:
                        for s in handler:
                            s(robot.karel, robot)
                    candidates_reward += robot.end_acc()
                    #candidates_reward += robot.execute_single_action(k_end(dummy=True))
                
                reward = candidates_reward / len(robots.robots)

            self.mask = COND_INDEX
            
        return reward, self.mask

    

class EnvEncoder(nn.Module):
    def __init__(self, h, w):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                        kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                        kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 512),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        avg_x = torch.zeros(1, x.shape[1])
        for i in range(batch_size):
            avg_x += x[i, :]

        return avg_x / batch_size