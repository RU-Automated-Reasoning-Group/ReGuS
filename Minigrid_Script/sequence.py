import pdb
import copy
import random
import numpy as np
import gc
import pickle
import argparse

import search
import minigrid_base_dsl
import mcts_search
import minigrid_implement.dsl
import minigrid
minigrid.register_minigrid_envs()

def start_sequence(
    args, envs, starting_idx=None, library_check_point=None, library_dict_check_point=None
):
    if starting_idx is not None:
        # resume mode
        pass
    else:
        starting_idx = 0

    if library_check_point is not None:
        with open(library_check_point, "rb") as library_f:
            library = pickle.load(library_f)
    else:
        library = []

    if library_dict_check_point is not None:
        with open(library_dict_check_point, "rb") as library_dict_f:
            library_dict = pickle.load(library_dict_f)
    else:
        library_dict = {}

    # pdb.set_trace()
    minigrid_base_dsl.set_library(library, library_dict)

    for idx in range(starting_idx, len(envs)):
        env = envs[idx]

        if idx == starting_idx:
            # set environment for the first env in this run
            minigrid_base_dsl.set_env(env)
            minigrid_base_dsl.set_action_dict(env)
            minigrid_base_dsl.set_cond_dict(env)

        # search for a correct program
        post_cond = lambda x: True # no post condition
        iters_per_sketch = 0
        step_limit_per_sketch = 1e9
        allow_action_first = False
        if idx == 0:
            p = minigrid_base_dsl.Program().expand()[0].expand()[5].expand()[0].expand()[1].expand()[1].expand()[0].expand()[0]
            def pt(program):
                new_p = copy.deepcopy(program)
                new_p.parameterize("goal", "locked_door")
                new_p2 = copy.deepcopy(new_p)
                from minigrid_implement.robot import MiniGridRobot
                r = MiniGridRobot("MiniGrid-LockedRoom-v0", seed=35)
                r.force_execution = True
                r2 = MiniGridRobot("MiniGrid-DoorKey-8x8-v0", seed=2)
                r2.force_execution = True
                pdb.set_trace()
                new_p.execute(r)
                new_p2.execute(r2)
                if r.env.env.env.front_is_locked_door() and r2.env.env.env.front_is_locked_door():
                    return True
                return False
            post_cond = pt
            iters_per_sketch = 1000
            step_limit_per_sketch = 300000
        elif idx == 1:
            iters_per_sketch = 500
            p = None
        elif idx == 2:
            iters_per_sketch = 50
            p = None
        elif idx == 3:
            iters_per_sketch = 100
            p = None
            def pt(program):
                get_key = copy.deepcopy(program)
                get_key.parameterize("goal", "key")
                get_locked_door = copy.deepcopy(program)
                get_locked_door.parameterize("goal", "locked_door")
                from minigrid_implement.robot import MiniGridRobot
                r = MiniGridRobot("MiniGrid-LockedRoom-v0", seed=35)
                r.force_execution = True
                get_key.execute(r)
                pdb.set_trace()
                r.execute_single_action(minigrid_implement.dsl.k_action("pickup"))
                r.env.env.env.render("1.png")
                r.active = True
                get_locked_door.execute(r)
                r.env.env.env.render("2.png")
                if r.env.env.env.front_is_locked_door():
                    return True
                return False
            post_cond = pt
        elif idx == 4:
            iters_per_sketch = 7000
            p = library_dict["get"]
            def pt(program):
                get_locked_door = copy.deepcopy(program)
                get_locked_door.parameterize("goal", "locked_door")
                from minigrid_implement.robot import MiniGridRobot
                r = MiniGridRobot("MiniGrid-DoorKey-8x8-v0", seed=1)
                r.force_execution = True
                get_locked_door.execute(r)
                r.env.env.env.render("1.png")
                if r.env.env.env.front_is_locked_door():
                    return True
                return False
            post_cond = pt
        elif idx == 5:
            p = library_dict["LK_get"]
        elif idx == 6:
            # put near
            allow_action_first = True
            p = None
            iters_per_sketch = 1000
        elif idx == 7:
            # pdb.set_trace()
            # p = minigrid_base_dsl.Program().expand()[1].expand()[25].expand()[0].expand()[0]
            p = minigrid_base_dsl.Program().expand()[1].expand()[1].expand()[0].expand()[0]
            iters_per_sketch = 300
            allow_action_first = True
            # step_limit_per_sketch = 300000
        # elif idx == 6:
        #     # p = (
        #     #     minigrid_base_dsl.Program()
        #     #     .expand()[1]
        #     #     .expand()[23]
        #     #     .expand()[0]
        #     #     .expand()[0]
        #     # )
        #     p = (
        #         minigrid_base_dsl.Program()
        #         .expand()[1]
        #         .expand()[23]
        #         .expand()[0]
        #         .expand()[2]
        #         .expand()[25]
        #         .expand()[0]
        #         .expand()[0]
        #     )
        else:
            p = None

        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        seed = args.seed
        more_seeds = [i for i in range(args.seed * 10, args.seed * 10 + 10)]
        # more_seeds = [i for i in range(0, 50)]
        eval_seeds = [i for i in range(1, 1000)]
        make_video = False
        # if idx in [0, 6]:
        node = search.Node(
            sketch=p,
            task=env,
            seed=seed,
            more_seeds=more_seeds,
            eval_seeds=eval_seeds,
            max_search_iter=700000,
            max_structural_cost=20,
            shuffle_actions=True,
            found_one=True,
            make_video=make_video,
            post_condition=post_cond,
            step_limit_per_sketch=step_limit_per_sketch,
            allow_action_first=allow_action_first
        )

        try:
            if idx in [4, 5, 7]:
                success_prog = node.search()
            else:
                mcts_search.mcts_search(
                    1000,
                    iters_per_sketch,
                    task=env,
                    seed=seed,
                    more_seeds=more_seeds,
                    eval_seeds=eval_seeds,
                    lib_actions=library,
                    post_condition=post_cond,
                    allow_action_first=allow_action_first
                )
                gc.collect()
        except search.SUCC as succ:
            success_prog = succ.p
        # execute the program and update its ABS_STATE further

        if idx == 2:
            success_prog.stmts = success_prog.stmts[:2]
            success_prog.stmts.extend(copy.deepcopy(library_dict["RC_get"].stmts))
            success_prog.reset()
            # pdb.set_trace()

        # change the global variable for the next environment
        print("setting DSL for next environment")
        next_env = envs[idx + 1]
        minigrid_base_dsl.set_env(next_env)
        minigrid_base_dsl.set_action_dict(next_env)
        minigrid_base_dsl.set_cond_dict(next_env)

        # copy the old ABS_STATE
        minigrid_implement.dsl.SEARCH_STATUS = False
        print("copying over ABS STATE")
        success_prog.copy_over_abs_state()
        # print(success_prog.to_string_verbose())

        print("executing and updating")
        node.execute_and_update(success_prog)
        print(success_prog.to_string_verbose())
        # pdb.set_trace()
        minigrid_implement.dsl.SEARCH_STATUS = True

        # add to the library
        library.append(success_prog)
        if idx == 0:
            library_dict["RC_get"] = success_prog
        elif idx == 1:
            library_dict["Lava_get"] = success_prog
        elif idx == 2:
            pdb.set_trace()
            library_dict["get"] = success_prog
        elif idx == 3:
            library_dict["get"] = success_prog
            key_get = copy.deepcopy(success_prog)
            key_get.parameterize("goal", "key")

            locked_door_get = copy.deepcopy(success_prog)
            locked_door_get.parameterize("goal", "locked_door")

            library.append(key_get)
            library.append(locked_door_get)

            library_dict["get_key"] = key_get
            library_dict["get_locked_door"] = locked_door_get
        elif idx == 4:
            library_dict["LK_get"] = success_prog
        elif idx == 5:
            library_dict["DK_get"] = success_prog

            ball_get = copy.deepcopy(success_prog)
            ball_get.parameterize("goal", "ball")

            library.append(ball_get)
            library_dict["get_ball"] = ball_get
        elif idx == 6:
            library_dict["put_near"] = success_prog

        # update the library to the DSL module
        minigrid_base_dsl.set_library(library, library_dict)

        print("saving check point")
        with open(f"library_ckpt_{idx + 1}", "wb") as library_f:
            pickle.dump(library, library_f)

        with open(f"library_dict_ckpt_{idx + 1}", "wb") as library_dict_f:
            pickle.dump(library_dict, library_dict_f)

        print(f"total interaction is {minigrid_implement.dsl.search_counter}")
        print("============ Moving to next env ================")
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", action="store", default=None)
    parser.add_argument("--library", action="store", default=None)
    parser.add_argument("--library_dict", action="store", default=None)
    parser.add_argument("--seed", type=int, action="store", required=True)
    parser.add_argument("--random_seed", type=int, action="store", required=True)

    args = parser.parse_args()
    # pdb.set_trace()
    envs = [
        "MiniGrid-RandomCrossingS11N5-v0",
        "MiniGrid-RandomLavaCrossingS11N5-v0",
        "MiniGrid-MultiRoomNoDoor-N6-v0",
        "MiniGrid-MultiRoom-N6-v0",
        "MiniGrid-LockedRoom-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-PutNearTwoRoom-v0",
        "MiniGrid-UnlockPickup-v0",
    ]

    if args.idx is None:
        start_sequence(args, envs)
    else:
        start_sequence(args, envs, int(args.idx), args.library, args.library_dict)