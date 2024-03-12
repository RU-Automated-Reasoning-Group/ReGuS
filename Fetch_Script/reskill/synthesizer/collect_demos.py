import argparse
import numpy as np
from reskill.utils.controllers.pick_and_place_controller import (
    get_pick_and_place_control,
)
from reskill.utils.controllers.push_controller import get_push_control
from reskill.utils.controllers.hook_controller import get_hook_control
import gym
import gymnasium
from tqdm import tqdm
from reskill.utils.general_utils import AttrDict
import reskill.rl.envs
from tqdm import tqdm
import random
import os
from perlin_noise import PerlinNoise


class CollectDemos:
    def __init__(self, dataset_name, num_trajectories=5, task="block"):
        self.task = task
        # self.dataset_dir = "../dataset/" + dataset_name + "/"
        # os.makedirs(self.dataset_dir, exist_ok=True)
        # self.save_dir = "../dataset/" + dataset_name + "/" + "demos.npy"
        self.num_trajectories = num_trajectories
        if self.task == "hook":
            # self.env = gymnasium.make('FetchHook-v0', render_mode='rgb_array')
            self.env = gym.make("FetchHookOptimized-v0", seed = 0)
        else:
            # On FetchCleanUp-v0, observe failures due to collision with container edges
            # with get_pick_and_place_control
            self.env = gym.make("FetchPlaceMultiGoal-v0", seed = 9)
            # self.env = gym.make("FetchCleanUp-v0", seed = 0)

    def get_obs(self, obs):
        return np.concatenate([obs["observation"], obs["desired_goal"]])

    def collect(self):
        print("Collecting demonstrations...")
        for i in tqdm(range(self.num_trajectories)):
            if "gym." in str(type(self.env)):
                obs = self.env.reset()
            elif "gymnasium." in str(type(self.env)):
                obs, _ = self.env.reset()
            else:
                assert False
            done1 = False
            done2 = False
            actions = []
            observations = []
            terminals = []

            if self.task == "block":
                controller = random.choice([get_pick_and_place_control])
            else:
                controller = get_hook_control

            idx = 0

            while not done1 and not done2:
                o = self.get_obs(obs)
                observations.append(o)

                idx += 1

                action, success = controller(obs)

                actions.append(action)

                ret = self.env.step(action)
                if "gym." in str(type(self.env)):
                    obs, rd, done1, info = ret
                elif "gymnasium." in str(type(self.env)):
                    obs, rd, done1, done2, info = ret
                else:
                    assert False

                print(f"reward is {rd}")
                terminals.append(success)

                self.env.render()
                if success:
                    break

            print(f"success = {success} and info = {info} and rd = {rd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trajectories", type=int, default=10)
    parser.add_argument("--task", type=str, default="block", choices=["block", "hook"])
    args = parser.parse_args()

    dataset_name = "fetch_" + args.task + "_" + str(args.num_trajectories)
    collector = CollectDemos(
        dataset_name=dataset_name,
        num_trajectories=args.num_trajectories,
        task=args.task,
    )
    collector.collect()
