# Program Synthesis of Intelligent Agent from Rewards


 Program Synthesis from Reward (PBR) is a programming-by-reward paradigm to unlock the potential of program synthesis to overcome the exploration challenges. We develop a novel hierarchical synthesis algorithm with decomposed search space for loops, on-demand synthesis of conditional statements, and curriculum synthesis for procedure calls, to effectively compress the search space for long-horizon, multi-stage, and procedural robot-control tasks that are difficult to explore using deep RL techniques. Experiment results demonstrate that PBR significantly outperforms state-of-the-art deep RL algorithms and standard program synthesis baselines on challenging RL tasks including video games, autonomous driving, locomotion control, object manipulation, and embodied AI.


## Karel Environment

To evaluate the capability of loop sketch synthesis and on-demand conditional statement synthesis, we use a suite of discrete state and action environments with the "Karel The Robot" simulator taken from [Trivedi et al](https://arxiv.org/abs/2108.13643). In Karel environment, an agent navigates inside a 2D grid world with walls and modifies the world state by interaction with markers. These tasks feature randomly sampled agent positions, walls, markers, and goal configurations.

### Setup
- Required Installation
    ```
    Python 3.8+
    Numpy 1.23.5
    ```

### Low-level Loop Sketch Completion
Complete low-level loop program based on specific high level sketch. Entering ```karel``` directory, Sketch completion examples could be run by:
```
cd karel
sh pbr_sketch.sh
```
Seven Karel environments are provided in ```pbr_sketch.sh```. For more details of input parameters:
- **task_name**: name of karel environment chosen from topOff, cleanHouse, stairClimber, randomMaze, fourCorner, harvester, seeder and doorkey.
- **search_seed**: starting random seed for program search.
- **more_seed**: all random seeds for program search.
- **sub_goals**: intermediate reward goal provided in string with ```,``` as split character.  For example, string ```0.5,1``` for goals ```[0.5, 1.0]```. The default value is ```1```.
- **search_iter**: maximum iteration for program search. The default value is ```2000```.
- **max_stru_cost**: limit of structure depth. The default value is ```20```.
- **stru_weight**: weight for structure cost to calculate synthesis score. The default value is ```0.2```.

### MCTS Search
We apply Monte Carlo Tree Search (MCTS) method to search for candidate high quality loop sketch and complete low-level loop program based on selected loop sketch. Entering ```karel``` directory, MCTS search examples could be run by:
```
cd karel
sh pbr_mcts.sh
```
Our code will search for program from root sketch ```S;``` until a sucess program is found or limited time (20 hours) is reached.

### Examples
- **Program search for Karel doorkey.**

    Synthesized Program:

    ```
    WHILE(not (markers_present)) { 
        IF(not (left_is_clear)) { 
            move
        }  
        IF(markers_present) { 
            IF(front_is_clear) { 
                turn_right
            } 
        }  
        IF(not (markers_present)) { 
            turn_right
        }  
        move
    } ; 
    pick_marker 
    WHILE(not (markers_present)) { 
        IF(front_is_clear) { 
            turn_left
        }  
        IF(not (front_is_clear)) { 
            turn_right
        } 
        move
    } ; 
    put_marker
    ; END
    ```

<figure>
<p align='center'>
<img src='./karel/figs/doorkey.gif'  height=200 width=200>
<img src='./karel/figs/doorkey_1.gif'  height=200 width=200>
<img src='./karel/figs/doorkey_2.gif'  height=200 width=200>
<center>
<hr><br>
</figure>

- **Program search for Karel seeder.**

    Synthesized Program:

    ```
    WHILE(not (markers_present)) { 
        WHILE(not (markers_present)) { 
            put_marker 
            move
        } ; 
        IF(front_is_clear) { 
            IF(not (right_is_clear)) { 
                move
            } 
        }  
        IF(front_is_clear) { 
            move
        }  
        IF(front_is_clear) { 
            move
        }  
        turn_left 
        pick_marker
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./karel/figs/seeder.gif'  height=200 width=200>
<img src='./karel/figs/seeder_1.gif'  height=200 width=200>
<img src='./karel/figs/seeder_2.gif'  height=200 width=200>
<center>
<hr><br>
</figure>


- **Program search for Karel topOff.**

    Synthesized Program:

    ```
    WHILE(front_is_clear) { 
        IF(markers_present) { 
            put_marker
        }  
        move
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./karel/figs/topOff.gif'  height=200 width=200>
<img src='./karel/figs/topOff_1.gif'  height=200 width=200>
<img src='./karel/figs/topOff_2.gif'  height=200 width=200>
<center>
<hr><br>
</figure>


- **Program search for Karel cleanHouse.**

    Synthesized Program:

    ```
    WHILE(not (markers_present)) { 
        WHILE(not (markers_present)) { 
            IF(left_is_clear) { 
                turn_left
            }  
            IF(not (front_is_clear)) { 
                turn_right
            }  
            move
        } ; 
        pick_marker
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./karel/figs/cleanHouse.gif'  height=200 width=300>
<img src='./karel/figs/cleanHouse_1.gif'  height=200 width=300>
<center>
<hr><br>
</figure>


- **Program search for Karel stairClimber.**

    Synthesized Program:

    ```
    WHILE(not (front_is_clear)) { 
        turn_left 
        move 
        turn_right 
        move
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./karel/figs/stairClimber.gif'  height=200 width=200>
<img src='./karel/figs/stairClimber_1.gif'  height=200 width=200>
<img src='./karel/figs/stairClimber_2.gif'  height=200 width=200>
<center>
<hr><br>
</figure>

- **Program search for Karel randomMaze.**

    Synthesized Program:

    ```
    WHILE(not (markers_present)) { 
        IF(right_is_clear) { 
            turn_right
        }  
        IF(not (front_is_clear)) { 
            turn_left
        } 
        move
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./karel/figs/randomMaze.gif'  height=200 width=200>
<img src='./karel/figs/randomMaze_1.gif'  height=200 width=200>
<img src='./karel/figs/randomMaze_2.gif'  height=200 width=200>
<center>
<hr><br>
</figure>


- **Program search for Karel fourCorner.**

    Synthesized Program:

    ```
    WHILE(left_is_clear) { 
        IF(not (front_is_clear)) { 
            put_marker 
            turn_left
        }  
        move
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./karel/figs/fourCorners.gif'  height=300 width=300>
<center>
<hr><br>
</figure>

- **Program search for Karel Harvester**

    Synthesized Program:

    ```
    WHILE(left_is_clear) { 
        IF(not (markers_present)) { 
            IF(right_is_clear) { 
                move
            }  
            IF(not (markers_present)) { 
                turn_left
            } 
        } 
        ELSE { 
            pick_marker
        } 
        IF(markers_present) { 
            turn_right
        } 
        move
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./karel/figs/harvester.gif'  height=300 width=300>
<center>
<hr><br>
</figure>

## MiniGrid Environment

To demonstrate the library learning feature of our approach, we use five environments that are of increaing difficulty from the [MiniGrid](https://github.com/Farama-Foundation/Minigrid) repository. The environments used are shown below

![5 environments](minigrid/figs/envs.png)

To run our approach and solve this 5 environments in order, please use the following commands

```
cd minigrid
python3 sequence.py
```

### Examples

Here we show the program synthesized for each of the 5 environemtns.

- **Program search for RandomCrossing**

    Synthesized Program:
    ```
    WHILE(not (goal_on_right)) { 
        IF(left_is_clear) { turn_left}  
        IF(goal_present) { return}  
        IF(not (front_is_clear)) { turn_right}  
        move
    } ; 
    turn_right 
    WHILE(not (goal_present)) { 
        move
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./minigrid/figs/RandomCrossing.gif'  height=200 width=200>
<img src='./minigrid/figs/RandomCrossing_1.gif'  height=200 width=200>
<img src='./minigrid/figs/RandomCrossing_2.gif'  height=200 width=200>
<center>
<hr><br>
</figure>

- **Program search for LavaCrossing**

    Synthesized Program:
    ```
    WHILE(not (goal_on_right)) { 
        IF(left_is_clear) { turn_left}  
        IF(goal_present) { return}  
        IF(not (front_is_clear)) { turn_right}  
        IF(front_is_lava) { turn_right}  
        move
    } ; 
    turn_right 
    WHILE(not (goal_present)) { 
        move
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./minigrid/figs/LavaCrossing.gif'  height=200 width=200>
<img src='./minigrid/figs/LavaCrossing_1.gif'  height=200 width=200>
<img src='./minigrid/figs/LavaCrossing_2.gif'  height=200 width=200>
<center>
<hr><br>
</figure>


- **Program search for MultiRoom**

    Synthesized Program:
    ```
    WHILE(front_is_clear) { 
        IF(front_is_closed_door) { turn_left}  
        move
    } ; 
    turn_right 
    WHILE(not (goal_on_right)) { 
        IF(left_is_clear) { turn_left}  
        IF(goal_present) { return}  
        IF(front_is_closed_door) { toggle}  
        IF(not (front_is_clear)) { turn_right}  
        move
    } ; 
    turn_right 
    WHILE(not (goal_present)) { 
        move
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./minigrid/figs/MultiRoom.gif'  height=200 width=200>
<img src='./minigrid/figs/MultiRoom_1.gif'  height=200 width=200>
<img src='./minigrid/figs/MultiRoom_2.gif'  height=200 width=200>
<center>
<hr><br>
</figure>

- **Program search for LockedRoom**

    Synthesized Program:
    ```
    WHILE(front_is_clear) { 
        IF(front_is_closed_door) { turn_left}  
        move
    } ; 
    turn_right 
    WHILE(not (goal_on_right)) { 
        IF(left_is_clear) { turn_left}  
        IF(front_is_key) { pickup}  
        IF(goal_present) { return}  
        IF(front_is_closed_door) { 
            IF(front_is_locked_door) { 
                IF(not (has_key)) { 
                    get_key pickup
                } 
                get_locked_door
            }  
            toggle
        }  
        IF(not (front_is_clear)) { turn_right}  
        move
    } ;
    turn_right
    WHILE(not (goal_present)) { 
        move
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./minigrid/figs/LockedRoom.gif'  height=200 width=200>
<img src='./minigrid/figs/LockedRoom_1.gif'  height=200 width=200>
<img src='./minigrid/figs/LockedRoom_2.gif'  height=200 width=200>
<center>
<hr><br>
</figure>

- **Program search for DoorKey**

    Synthesized Program:
    ```
    WHILE(front_is_clear) { 
        IF(front_is_closed_door) { turn_left}  
        move
    } ; 
    turn_right 
    WHILE(not (goal_on_right)) { 
        IF(left_is_clear) { turn_left}  
        IF(front_is_key) { pickup}  
        IF(goal_present) { return}  
        IF(front_is_closed_door) { 
            IF(front_is_locked_door) { 
                IF(not (has_key)) { 
                    get_key pickup
                }  
                get_locked_door
            }  
            toggle
        }  
        IF(not (front_is_clear)) { turn_right}  
        move
    } ; 
    turn_right
    WHILE(not (goal_present)) { 
        move
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./minigrid/figs/DoorKey.gif'  height=200 width=200>
<img src='./minigrid/figs/DoorKey_1.gif'  height=200 width=200>
<img src='./minigrid/figs/DoorKey_2.gif'  height=200 width=200>
<center>
<hr><br>
</figure>