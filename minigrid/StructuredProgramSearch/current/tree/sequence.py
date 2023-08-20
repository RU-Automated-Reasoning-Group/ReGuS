import search
import minigrid_base_dsl
import copy
import mcts_search
import random
import numpy as np
import gc
import pickle
import argparse
import Minigrid


def start_sequence(
    envs, starting_idx=None, library_check_point=None, library_dict_check_point=None
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

    minigrid_base_dsl.set_library(library, library_dict)

    for idx in range(starting_idx, len(envs)):
        env = envs[idx]

        if idx == starting_idx:
            # set environment for the first env in this run
            minigrid_base_dsl.set_env(env)
            minigrid_base_dsl.set_action_dict(env)
            minigrid_base_dsl.set_cond_dict(env)

        # search for a correct program
        if idx == 0:
            p = (
                minigrid_base_dsl.Program()
                .expand()[0]
                .expand()[9]
                .expand()[0]
                .expand()[2]
                .expand()[11]
                .expand()[0]
                .expand()[0]
            )
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

        random.seed(3)
        np.random.seed(3)
        seed = 0
        more_seeds = [i for i in range(1, 50)]
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
        )

        try:
            if idx in []:
                success_prog = node.search()
            else:
                mcts_search.mcts_search(
                    1000,
                    2000,
                    task=env,
                    seed=seed,
                    more_seeds=more_seeds,
                    eval_seeds=eval_seeds,
                    lib_actions=library,
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
        print("copying over ABS STATE")
        success_prog.copy_over_abs_state()
        # print(success_prog.to_string_verbose())

        print("executing and updating")
        node.execute_and_update(success_prog)
        print(success_prog.to_string_verbose())
        # pdb.set_trace()

        # add to the library
        library.append(success_prog)
        if idx == 0:
            library_dict["RC_get"] = success_prog
        elif idx == 1:
            library_dict["Lava_get"] = success_prog
        elif idx == 2:
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

        # update the library to the DSL module
        minigrid_base_dsl.set_library(library, library_dict)

        print("saving check point")
        with open(f"library_ckpt_{idx + 1}", "wb") as library_f:
            pickle.dump(library, library_f)

        with open(f"library_dict_ckpt_{idx + 1}", "wb") as library_dict_f:
            pickle.dump(library_dict, library_dict_f)

        print("============ Moving to next env ================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", action="store", default=None)
    parser.add_argument("--library", action="store", default=None)
    parser.add_argument("--library_dict", action="store", default=None)
    args = parser.parse_args()
    envs = [
        "MiniGrid-RandomCrossingS11N5-v0",
        "MiniGrid-RandomLavaCrossingS11N5-v0",
        "MiniGrid-MultiRoomNoDoor-N6-v0",
        "MiniGrid-MultiRoom-N6-v0",
        "MiniGrid-LockedRoom-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-UnlockPickup-v0",
    ]

    if args.idx is None:
        start_sequence(envs)
    else:
        start_sequence(envs, int(args.idx), args.library, args.library_dict)

    # library = []
    # library_dict = {}
    # for idx, env in enumerate(envs):
    #     # set environment name and DSL and ABS_STATE for first env
    #     if idx == 0:
    #         minigrid_base_dsl.set_env(env)
    #         minigrid_base_dsl.set_action_dict(env)
    #         minigrid_base_dsl.set_cond_dict(env)

    #     # search for a correct program
    #     if idx == 0:
    #         p = (
    #             minigrid_base_dsl.Program()
    #             .expand()[0]
    #             .expand()[9]
    #             .expand()[0]
    #             .expand()[2]
    #             .expand()[11]
    #             .expand()[0]
    #             .expand()[0]
    #         )
    #     elif idx == 6:
    #         # p = (
    #         #     minigrid_base_dsl.Program()
    #         #     .expand()[1]
    #         #     .expand()[23]
    #         #     .expand()[0]
    #         #     .expand()[0]
    #         # )
    #         p = (
    #             minigrid_base_dsl.Program()
    #             .expand()[1]
    #             .expand()[23]
    #             .expand()[0]
    #             .expand()[2]
    #             .expand()[25]
    #             .expand()[0]
    #             .expand()[0]
    #         )

    #     random.seed(3)
    #     np.random.seed(3)
    #     seed = 0
    #     more_seeds = [i for i in range(1, 50)]
    #     eval_seeds = [i for i in range(1, 1000)]
    #     make_video = False
    #     if idx in [0, 6]:
    #         node = search.Node(
    #             sketch=p,
    #             task=env,
    #             seed=seed,
    #             more_seeds=more_seeds,
    #             eval_seeds=eval_seeds,
    #             max_search_iter=700000,
    #             max_structural_cost=20,
    #             shuffle_actions=True,
    #             found_one=True,
    #             make_video=make_video,
    #         )

    #     try:
    #         if idx in [0, 6]:
    #             success_prog = node.search()
    #         else:
    #             mcts_search.mcts_search(
    #                 100,
    #                 1000,
    #                 task=env,
    #                 seed=seed,
    #                 more_seeds=more_seeds,
    #                 eval_seeds=eval_seeds,
    #                 lib_actions=library,
    #             )
    #             gc.collect()
    #     except search.SUCC as succ:
    #         success_prog = succ.p
    #     # execute the program and update its ABS_STATE further

    #     if idx == 2:
    #         success_prog.stmts = success_prog.stmts[:2]
    #         success_prog.stmts.extend(copy.deepcopy(library_dict["RC_get"].stmts))
    #         success_prog.reset()
    #         # pdb.set_trace()

    #     # change the global variable for the next environment
    #     print("setting DSL for next environment")
    #     next_env = envs[idx + 1]
    #     minigrid_base_dsl.set_env(next_env)
    #     minigrid_base_dsl.set_action_dict(next_env)
    #     minigrid_base_dsl.set_cond_dict(next_env)

    #     # copy the old ABS_STATE
    #     print("copying over ABS STATE")
    #     success_prog.copy_over_abs_state()
    #     # print(success_prog.to_string_verbose())

    #     print("executing and updating")
    #     node.execute_and_update(success_prog)
    #     print(success_prog.to_string_verbose())
    #     # pdb.set_trace()

    #     # add to the library
    #     library.append(success_prog)
    #     if idx == 0:
    #         library_dict["RC_get"] = success_prog
    #     elif idx == 1:
    #         library_dict["Lava_get"] = success_prog
    #     elif idx == 2:
    #         library_dict["get"] = success_prog
    #     elif idx == 3:
    #         library_dict["get"] = success_prog
    #         key_get = copy.deepcopy(success_prog)
    #         key_get.parameterize("goal", "key")

    #         locked_door_get = copy.deepcopy(success_prog)
    #         locked_door_get.parameterize("goal", "locked_door")

    #         library.append(key_get)
    #         library.append(locked_door_get)

    #         library_dict["get_key"] = key_get
    #         library_dict["get_locked_door"] = locked_door_get
    #     elif idx == 4:
    #         library_dict["LK_get"] = success_prog
    #     elif idx == 5:
    #         library_dict["DK_get"] = success_prog

    #         ball_get = copy.deepcopy(success_prog)
    #         ball_get.parameterize("goal", "ball")

    #         library.append(ball_get)
    #         library_dict["get_ball"] = ball_get

    #     # update the library to the DSL module
    #     minigrid_base_dsl.set_library(library, library_dict)

    #     print("saving check point")
    #     with open(f"library_ckpt_{idx + 1}", "wb") as library_f:
    #         pickle.dump(library, library_f)

    #     with open(f"library_dict_ckpt_{idx + 1}", "wb") as library_dict_f:
    #         pickle.dump(library_dict, library_dict_f)

    #     print("============ Moving to next env ================")
