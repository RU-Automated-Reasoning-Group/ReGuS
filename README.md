# Program Synthesis of Intelligent Agent from Rewards


 Program Synthesis from Reward (PBR) is a programming-by-reward paradigm to unlock the potential of program synthesis to overcome the exploration challenges. We develop a novel hierarchical synthesis algorithm with decomposed search space for loops, on-demand synthesis of conditional statements, and curriculum synthesis for procedure calls, to effectively compress the search space for long-horizon, multi-stage, and procedural robot-control tasks that are difficult to explore using deep RL techniques. Experiment results demonstrate that PBR significantly outperforms state-of-the-art deep RL algorithms and standard program synthesis baselines on challenging RL tasks including video games, autonomous driving, locomotion control, object manipulation, and embodied AI.


## Karel Environment

---

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
- **task_name**: name of karel environment choicen from topOff, cleanHouse, stairClimber, randomMaze, fourCorner, harvester, seeder and doorkey.
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

![](https://github.com/RU-Automated-Reasoning-Group/PBR/tree/main/karel/figs/doorkey.gif)


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
<img src='./figs/seeder.gif'  height=300 width=300>
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
<img src='./figs/topoff.gif'  height=300 width=300>
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
<img src='./figs/cleanHouse.gif'  height=300 width=300>
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
<img src='./figs/stairClimber.gif'  height=300 width=300>
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
<img src='./figs/randomMaze.gif'  height=300 width=300>
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
<img src='./figs/fourCorners.gif'  height=300 width=300>
<center>
<hr><br>
</figure>

- **Program search for Karel Harvester**

Synthesized Program:

    ```
    WHILE(left_is_clear) { 
        WHILE(markers_present) { 
            pick_marker 
            IF(not (front_is_clear)) { 
                turn_left
            }  
            move
        } ; 
        IF(right_is_clear) { 
            move
        }  
        IF(markers_present) {
            turn_right
        } 
        ELSE { 
            turn_left
        } 
        move
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./figs/harvester.gif'  height=300 width=300>
<center>
<hr><br>
</figure>
