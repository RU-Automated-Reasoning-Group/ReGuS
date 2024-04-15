# Fetch-Pick&Place and Fetch-Hook

To run the code for fetch environments, use conda env regus 2 by running
```
conda activate regus2
```

Set environment variable in bash
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
```

## 1.  Synthesizing a new program for Pick&Place environment
There are two required arguments. Seed is used to specify different environment initial configuration. Random_seed is is used to control the randomness in the program


seed and random_seed can be changed accordingly.

The estimated running time is 15 minutes. The synthesized program can be found in file "pick.txt" after the synthesis terminates. The output of this program is a is a detailed log showing how the result program is synthesized.
```
python3 search.py --seed 0 --random_seed 0
```

One example program is
```
WHILE(not (block_at_goal)) {
        IF(block_below_gripper) {
            IF(gripper_are_open) {
                IF(block_inside_gripper) {
                    close_gripper
                } ELSE {
                    move_down
                }
            } ELSE {
                IF(block_inside_gripper) {
                    move_to_goal
                } ELSE {
                    open_gripper
                }
            }
        } ELSE {
            move_to_block
        }
    idle
} ;; END
```


## 2. Synthesizing a new program for Hook environment

The estimated running time is 2.5 minutes. The synthesized program can be found in file "hook.txt" after the synthesis terminates.
```
python3 search_hook.py --seed 0 --random_seed 0
```

One example program is
```
WHILE(not (block_at_goal)) {
        IF(hook_grasped) {
            IF(hook_aligned) {
                sweep
            } ELSE {
                align
            }
        } ELSE {
            pick_up_hook
        }
    idle
} ;; END
```

## 3. Generate Plot (Figure 22 b&c)
```
python3 generate_plot.py
```

The output are two PDF files which are similar curves for ReGus as in Fig 22 (b) & (c). The trailing horizontal line is missing because ReGuS terminates once it finds the best program.

## 4. Code Structure
- `reskill` folder contains the environment files
- `programskill` folder contains the actual implementation of the low level actions and predicates that directly interact with the low-level environment.
- `robot_dsl.py` & `robot_hook_dsl.py` specify the action and predicate sets used for each of the environments. These two files also implement the Domain Specific Language (DSL) used by ReGuS. These two files should be considered as the interface of ReGuS that users will need to change for new environments or test new actions.
- `search.py` contains the partial program executor for ReGuS DSL and implement the main logic of ReGuS.

## 5. Configuring Predicates Used for Synthesis

1. ReGuS predicates interface

    The predicates used for synthesis for environment can be configured by adjusting the `COND_DICT` dictionary defined at line 39 - 80 of `robot.dsl` in the `Fetch_Script` folder. For example, the `block_at_goal` predicate and its negation can be defined as in the following code block. Users can comment these lines to disable this predicate. New predicates can be defined similarly. Experiments show that when removing exactly one predicate among `block_is_grasped`, `block_inside_gripper` and `gripper_open`, ReGuS can still find a successful program.

    ```
    COND_DICT = {
        "block_at_goal": k_cond(
            negation=False, cond=k_cond_without_not("block_at_goal")
        ),
        ...
        "not(block_at_goal)": k_cond(
            negation=True, cond=k_cond_without_not("block_at_goal")
        ),
    }
    ```

2. Environment predicate implementation

    However, the definition above is only the high level interface used by the ReGuS synthesis algorithm. The set of all states that satisfy `block_at_goal` is actually provided by the `Fetch-Pic&Place` environment. The `block_at_goal` function definition is located at line 554 - 557 at file `reskill/rl/envs/fetch_pick_and_place.py` in this folder. User can define new predicate by providing such a function that determines the set of states that the predicate is true or false.


## 6. Configuring Actions Used for Synthesis

1. ReGuS action interface

    The actions used for synthesis for environment can be configured by adjusting the `ACTION_DICT` dictionary defined at line 19 - 26 of `robot.dsl` in the `Fetch_Script` folder. For example, the `move_to_block` action is defined as in the following code block. Users can comment these lines to disable this action. New actions can be defined similarly.

    ```
    ACTION_DICT = {
        ...
        "move_to_block": k_action("move_to_block"),
        ...
    }
    ```

2. Action implementation

    Given a ReGuS action such as `move_to_block`, what effect it will have on the environment is defined in the `programskill/dsl.py` file. For example, line 96 - 110 is the implementation of `move_to_block`. This ReGuS action will find the position of the block and generate a low level action to move the gripper towards this location. New actions can be defined similarly by accessing the `k.env.obs` as the state from the environment.
