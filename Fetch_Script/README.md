# Fetch-Pick&Place and Fetch-Hook

## 1.  Synthesizing a new program for Pick&Place environment 
There are two required arguments. Seed is used to specify different environment initial configuration. Random_seed is is used to control the randomness in the program


seed and random_seed can be changed accordingly.

The estimated running time is 15 minutes. The synthesized program can be found in file "pick.txt" after the synthesis terminates.
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

## 3. Generate Plot
```
python3 generate_plot.py
```