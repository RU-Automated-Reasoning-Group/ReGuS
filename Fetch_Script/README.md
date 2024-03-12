# Fetch-Pick&Place and Fetch-Hook

To run the code for fetch environments, use conda env regus 2 by running
```
conda activate regus2
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

### Code Structure
- `reskill` folder contains the environment files
- `programskill` folder contains the actual implementation of the low level actions and predicates (what is underhood of actions or predicates from the ReGuS program) that directly interact with the environment
- `robot_dsl.py` & `robot_hook_dsl.py` specify the action and predicate sets used for each of the environments. These two files also implement the Domain Specific Language (DSL) used by ReGuS. These two files should be considered as the interface of ReGuS that users will most probabily change for new environents or test new actions.
- `search.py` contains the partial program executor for ReGuS DSL and implement the main logic of ReGuS, such as generate new actions within a program or synthesize if/ifelse when necessary. 