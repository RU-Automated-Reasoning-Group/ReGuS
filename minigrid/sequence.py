import search
import minigrid_base_dsl
import copy
import mcts_search
import random
import numpy as np
import gc
import pickle
import argparse
import sys

import minigrid

import minigrid_implement.dsl
minigrid.register_minigrid_envs()


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

    import pdb
    # pdb.set_trace()
    # library_dict.pop("get")
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
        elif idx == 1:
            p = copy.deepcopy(library_dict["RC_get"])
        elif idx == 3:
            p = copy.deepcopy(library_dict["get"])
        elif idx == 4:
            p = copy.deepcopy(library_dict["get"])
        elif idx == 5:
            p = copy.deepcopy(library_dict["LK_get"])
        elif idx == 6:
            # import pdb
            # pdb.set_trace()
            # p = (
            #     minigrid_base_dsl.Program()
            #     .expand()[1]
            #     .expand()[19]
            #     .expand()[0]
            #     .expand()[0]
            # )
            p = minigrid_base_dsl.Program().expand()[1]
            p.stmts.pop(1)
            p.stmts.pop(1)
            p.stmts[0] = minigrid_base_dsl.C()
            # p = copy.deepcopy(library_dict["get_ball"])
        elif idx == 7:
            # import pdb
            # pdb.set_trace()
            p = minigrid_base_dsl.Program().expand()[1].expand()[23].expand()[0].expand()[0]
            # p = (
            #     minigrid_base_dsl.Program()
            #     .expand()[1]
            #     .expand()[25]
            #     .expand()[0]
            #     .expand()[0]
            # )
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

        import debug_program

        # p = debug_program.convert_program('WHILE(front_is_clear) { IF(front_is_closed_door) { turn_left}  IF(front_is_key) { turn_left}  move}  turn_right WHILE(not (goal_on_right)) { IF(left_is_clear) { turn_left}  IF(goal_present) { return}  IF(front_is_closed_door) { IF(not (front_is_clear)) { IF(not (has_key)) { IF(not (right_is_clear)) { turn_right}  IF(not (front_is_clear)) { get_key}  pickup}  get_locked_door}  toggle}  IF(front_is_key) { pickup}  IF(not (front_is_clear)) { turn_right}  move}  turn_right WHILE(not (goal_present)) { move}')
        random.seed(100)
        np.random.seed(100)
        seed = 0
        more_seeds = [i for i in range(1, 50)]
        # more_seeds.pop(more_seeds.index(6))
        # eval_seeds = [35, 408]
        eval_seeds = [i for i in range(1, 1000)]
        # tmp_list = [6, 14, 34, 54, 68, 70, 94, 102, 107, 110, 116, 117, 119, 131, 148, 150, 151, 155, 158, 163, 164, 165, 184, 200, 201, 210, 247, 253, 259, 265, 267, 272, 274, 286, 311, 312, 320, 324, 340, 350, 353, 366, 373, 380, 383, 386, 389, 415, 428, 430, 433, 444, 445, 447, 452, 457, 475, 486, 494, 523, 524, 544, 548, 554, 556, 558, 565, 581, 608, 610, 625, 630, 636, 649, 653, 654, 678, 679, 695, 701, 704, 713, 715, 727, 731, 732, 734, 745, 747, 750, 753, 759, 767, 788, 793, 798, 814, 816, 856, 858, 859, 865, 873, 881, 885, 894, 897, 907, 921, 929, 937, 941, 956, 971, 973, 988]
        # lst = []
        # for i in eval_seeds:
        #     if i not in tmp_list:
        #         lst.append(i)
        # eval_seeds = lst
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
        # import minigrid_implement.dsl
        # minigrid_implement.dsl.DSL_DEBUG = True
        # print(node.test_for_success_rate(p))
        # exit()

        # import debug_program
        # prog = debug_program.convert_program("  get_ball WHILE(has_key) { IF(not (left_is_clear)) { drop}  IF(not (left_is_clear)) { move}  turn_left} ")
        # node.test_for_success_rate(prog)
        # exit()
        
        try:
            if idx in [0, 1, 3, 4, 5, 6, 7]:
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
        elif idx == 6:
            library_dict["put_near"] = success_prog

        # update the library to the DSL module
        minigrid_base_dsl.set_library(library, library_dict)

        print("saving check point")
        with open(f"library_ckpt_{idx + 1}", "wb") as library_f:
            pickle.dump(library, library_f)

        with open(f"library_dict_ckpt_{idx + 1}", "wb") as library_dict_f:
            pickle.dump(library_dict, library_dict_f)

        print("============ Moving to next env ================")
        print(f"total number of interaction is {minigrid_implement.dsl.search_counter}")
        exit()


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
        "MiniGrid-PutNearTwoRoom-v0",
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
