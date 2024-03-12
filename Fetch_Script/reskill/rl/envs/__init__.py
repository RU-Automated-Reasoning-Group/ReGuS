from gym.envs.registration import register
import gymnasium
from reskill.rl.envs.envs import *
import sys
from functools import reduce


register(
    id="FetchStacking-v0",
    entry_point="reskill.rl.envs.envs:FetchStackingEnv",
    kwargs={"reward_info": "place_blue"},
    max_episode_steps=100,
)

register(
    id="FetchStackingMultiGoal-v0",
    entry_point="reskill.rl.envs.envs:FetchStackingEnv",
    kwargs={"reward_info": "place_blue", "use_fixed_goal": False},
    max_episode_steps=100,
)

register(
    id="FetchPyramidStack-v0",
    entry_point="reskill.rl.envs.envs:FetchStackingEnv",
    kwargs={
        "reward_info": "stack_red",
        "use_fixed_goal": False,
        "use_force_sensor": True,
    },
    max_episode_steps=50,
)

register(
    id="FetchPlaceMultiGoal-v0",
    entry_point="reskill.rl.envs.envs:FetchPlaceEnv",
    kwargs={"reward_info": "place", "use_fixed_goal": False, "use_force_sensor": True},
    max_episode_steps=100,
)

register(
    id="FetchPlaceABS-v0",
    entry_point="reskill.rl.envs.envs:FetchPlaceAbsEnv",
    kwargs={
        "seed": 0,
        "reward_info": "place",
        "use_fixed_goal": False,
        "use_force_sensor": True,
    },
    max_episode_steps=100,
)

register(
    id="FetchCleanUp-v0",
    entry_point="reskill.rl.envs.envs:FetchCleanUpEnv",
    kwargs={
        "reward_info": "cleanup_1block",
        "use_fixed_goal": True,
        "use_force_sensor": True,
    },
    max_episode_steps=50,
)

register(
    id="FetchSlipperyPush-v0",
    entry_point="reskill.rl.envs.envs:FetchSlipperyPushEnv",
    kwargs={"reward_info": "place", "use_fixed_goal": True},
    max_episode_steps=100,
)

register(
    id="FetchOptimized-v0",
    entry_point="reskill.rl.envs.fetch_optimized:FetchOptimizedEnv",
    max_episode_steps=50,
)

register(
    id="FetchPickAndPlace-v0",
    entry_point="reskill.rl.envs.fetch_pick_and_place:FetchPickAndPlace",
    max_episode_steps=100,
)

register(
    id="FetchPush-v0",
    entry_point="reskill.rl.envs.fetch_push:FetchPush",
    max_episode_steps=100,
)

register(
    id="FetchHookOptimized-v0",
    entry_point="reskill.rl.envs.fetch_hook:FetchHookOptimized",
    max_episode_steps=100,
)

print("============ environment imported ===============")

gymnasium.envs.registration.register(
    id="FetchHook-v0",
    entry_point="reskill.rl.envs.fetch_hook_env:FetchHookEnv",
    max_episode_steps=100,
)

gymnasium.envs.registration.register(
    id="FetchHookAbs-v0",
    entry_point="reskill.rl.envs.fetch_hook_env:FetchHookAbsEnv",
    max_episode_steps=100,
)

gymnasium.envs.registration.register(
    id="FetchComplexHook-v0",
    entry_point="reskill.rl.envs.complex_hook_env:ComplexHookEnv",
    max_episode_steps=100,
)

gymnasium.envs.registration.register(
    id="ComplexHookSingleObject-v0",
    entry_point="reskill.rl.envs.complex_hook_env:ComplexHookSingleObjectEnv",
    max_episode_steps=100,
)
