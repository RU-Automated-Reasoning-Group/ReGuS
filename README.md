# Program Synthesis of Intelligent Agent from Rewards


Deep reinforcement learning (RL) has led to encouraging successes in numerous challenging robotics applications. However, the lack of inductive biases to support logic deduction and generalization in the representation of a deep RL model causes it less effective in exploring complex long-horizon robot-control tasks with sparse reward signals. Existing program synthesis algorithms for RL problems inherit the same limitation, as they either adapt conventional RL algorithms to guide program search or synthesize robot-control programs to imitate an RL model. In this paper, we propose PBR, a programming-by-reward paradigm, to unlock the potential of program synthesis to overcome the exploration challenges. We develop a novel hierarchical synthesis algorithm with decomposed search space for loops, on-demand synthesis of conditional statements, and curriculum synthesis for procedure calls, to effectively compress the search space for long-horizon, multi-stage, and procedural robot-control tasks that are difficult to explore using deep RL techniques. Experiment results demonstrate that PBR significantly outperforms state-of-the-art deep RL algorithms and standard program synthesis baselines on challenging RL tasks including video games, autonomous driving, locomotion control, object manipulation, and embodied AI - operating home-assisted robots in complex household environments.


## Karel Environment

To evaluate the capability of loop sketch synthesis and on-demand conditional statement synthesis, we use a suite of discrete state and action environments with the "Karel The Robot" simulator, taken from [Trivedi et al](https://arxiv.org/abs/2108.13643). In these environments, an agent
navigates inside a 2D grid world with walls and modifies the world state by interaction with markers. These tasks feature randomly sampled agent positions, walls, markers, and goal configurations.


### Setup
- Installation
    ```
    Python 3.8+
    Numpy 1.23.5
    ```

### Low-level Loop Sketch Completion
Robot-control Program Synthesis on top of a loop sketch. Loop ketch completion examples can be run by:
```
cd karel
sh pbr_sketch.sh
```
Seven Karel environments are supported in ```pbr_sketch.sh```. The configurable parameters of the script are as follows:
- **task_name**: the name of the Karel environment which can be chosen from [topOff, cleanHouse, stairClimber, randomMaze, fourCorner, harvester, seeder, and doorkey].
- **search_seed**: the seed of the environment for program search.
- **more_seed**: the additional seeds for environments that a synthesized program must pass to be deemed correct.
- **search_iter**: the maximum number of program search iterations. The default value is ```2000```.
- **max_stru_cost**: the upper bound of program structure depth. The default value is ```20```.
- **stru_weight**: the weight of structure cost that is used to regularize reward-based search. The default value is ```0.2```.

### MCTS Search
We use a variant of Monte Carlo Tree Search (MCTS) to automatically discover loop sketches for robot-control program synthesis. MCTS-based fully automatic robot-control program search can be run by:
```
cd karel
sh pbr_mcts.sh
```
The synthesizer runs until a successful program is found or times-out after 2 hours.

### Examples
- **Program searched for Karel TopOff.**

    Synthesized Program:

    ```
    WHILE(frontIsClear()) { 
        IF(present(marker)) { 
            put(marker)
        }  
        move()
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


- **Program searched for Karel CleanHouse.**

    Synthesized Program:

    ```
    WHILE(not (present(marker))) { 
        WHILE(not (present(marker))) { 
            IF(leftIsClear()) { 
                turnLeft()
            }  
            IF(not (frontIsClear())) { 
                turnRight()
            }  
            move()
        } ; 
        pickUp(marker)
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


- **Program searched for Karel StairClimber.**

    Synthesized Program:

    ```
    WHILE(not (frontIsClear())) { 
        turnLeft() 
        move() 
        turnRight() 
        move()
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

- **Program searched for Karel RandomMaze.**

    Synthesized Program:

    ```
    WHILE(not (present(marker))) { 
        IF(right_is_clear) { 
            turnRight()
        }  
        IF(not (frontIsClear())) { 
            turnLeft()
        } 
        move()
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


- **Program searched for Karel FourCorner.**

    Synthesized Program:

    ```
    WHILE(leftIsClear()) { 
        IF(not (frontIsClear())) { 
            put(marker) 
            turnLeft()
        }  
        move()
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./karel/figs/fourCorners.gif'  height=300 width=300>
<center>
<hr><br>
</figure>

- **Program searched for Karel Harvester**

    Synthesized Program:

    ```
    WHILE(leftIsClear()) { 
        IF(not (present(marker))) { 
            IF(right_is_clear) { 
                move()
            }  
            IF(not (present(marker))) { 
                turnLeft()
            } 
        } 
        ELSE { 
            pickUp(marker)
        } 
        IF(present(marker)) { 
            turnRight()
        } 
        move()
    } ;
    ; END
    ```

<figure>
<p align='center'>
<img src='./karel/figs/harvester.gif'  height=300 width=300>
<center>
<hr><br>
</figure>

- **Program searched for Karel Doorkey.**

    Synthesized Program:

    ```
    WHILE(not (present(marker))) { 
        IF(not (leftIsClear())) { 
            move()
        }  
        IF(present(marker)) { 
            IF(frontIsClear()) { 
                turnRight()
            } 
        }  
        IF(not (present(marker))) { 
            turnRight()
        }  
        move()
    } ; 
    pickUp(marker) 
    WHILE(not (present(marker))) { 
        IF(frontIsClear()) { 
            turnLeft()
        }  
        IF(not (frontIsClear())) { 
            turnRight()
        } 
        move()
    } ; 
    put(marker)
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
    WHILE(not (present(marker))) { 
        WHILE(not (present(marker))) { 
            put(marker) 
            move()
        } ; 
        IF(frontIsClear()) { 
            IF(not (right_is_clear)) { 
                move()
            } 
        }  
        IF(frontIsClear()) { 
            move()
        }  
        IF(frontIsClear()) { 
            move()
        }  
        turnLeft() 
        pickUp(marker)
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


## MiniGrid Environment

We evaluate how PBR can expedite program synthesis for a stream of tasks with various complexity using [MiniGrid](https://github.com/Farama-Foundation/Minigrid), a collection of grid world environments with goal-oriented tasks, which are widely used to evaluate state-of-the-art reinforcement learning algorithms. The tasks involve solving different mazes and interacting with various objects such as goals (green squares), doors, keys, boxes, and walls. 

![5 environments](minigrid/figs/envs.png)

As the environments above increase in complexity as the number of objects to manipulate increases, PBR adds programs synthesized for a simpler environment as a new skill that can be reused to constitute sophisticated programs for more complex environments. For example, PBR adds the goal-reaching program synthesized for the third environment Multiroom as a new skill skill3 (obj) to our DSL (domain-specific language). This skill can then be reused as a new control action in the form of callable procedures such as skill3 (key) and skill3 (door). When synthesizing a program in LockedRoom, the agent can call skill3 (key) to grab the key if it faces a locked door. It can then return to the locked door via skill3 (door) to open it. New skills effectively compress the search space for the agent to explore long-horizon tasks. Using curriculum synthesis, PBR achieves the maximum 1.0 reward on all the 5 environments above evaluated over 5000 random seeds. The synthesized programs feature multiple sequential loops with deeply nested conditionals making them impossible to be synthesized by program enumeration.

To run PBR and solve the above 5 environments in the order of their complexity, use the following script:

```
cd minigrid
python3 sequence.py
```

### Examples
- **Program searched for RandomCrossing**

    Synthesized Program:
    ```
    WHILE(not (rightIs(goal))) { 
        IF(leftIsClear()) { turnLeft()}  
        IF(present(goal)) { return}  
        IF(not (frontIsClear())) { turnRight()}  
        move()
    } ; 
    turnRight() 
    WHILE(not (present(goal))) { 
        move()
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

- **Program searched for LavaCrossing**

    Synthesized Program:
    ```
    WHILE(not (rightIs(goal))) { 
        IF(leftIsClear()) { turnLeft()}  
        IF(present(goal)) { return}  
        IF(not (frontIsClear())) { turnRight()}  
        IF(present(lava)) { turnRight()}  
        move()
    } ; 
    turnRight() 
    WHILE(not (present(goal))) { 
        move()
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


- **Program searched for MultiRoom**

    Synthesized Program:
    ```
    WHILE(frontIsClear()) { 
        IF(present(closed_door)) { turnLeft()}  
        move()
    } ; 
    turnRight() 
    WHILE(not (rightIs(goal))) { 
        IF(leftIsClear()) { turnLeft()}  
        IF(present(goal)) { return}  
        IF(present(closed_door)) { toggle()}  
        IF(not (frontIsClear())) { turnRight()}  
        move()
    } ; 
    turnRight() 
    WHILE(not (present(goal))) { 
        move()
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

- **Program searched for LockedRoom**

    Synthesized Program:
    ```
    WHILE(frontIsClear()) { 
        IF(present(closed_door)) { turnLeft()}  
        move()
    } ; 
    turnRight() 
    WHILE(not (rightIs(goal))) { 
        IF(leftIsClear()) { turnLeft()}  
        IF(present(key)) { pickup()}  
        IF(present(goal)) { return}  
        IF(present(closed_door)) { 
            IF(present(locked_door)) { 
                IF(not (hasKey())) { 
                    get(key)
                    pickup()
                } 
                get(locked_door)
            }  
            toggle()
        }  
        IF(not (frontIsClear())) { turnRight()}  
        move()
    } ;
    turnRight()
    WHILE(not (present(goal))) { 
        move()
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

- **Program searched for DoorKey**

    Synthesized Program:
    ```
    WHILE(frontIsClear()) { 
        IF(present(closed_door)) { turnLeft()}  
        move()
    } ; 
    turnRight() 
    WHILE(not (rightIs(goal))) { 
        IF(leftIsClear()) { turnLeft()}  
        IF(present(key)) { pickup}  
        IF(present(goal)) { return}  
        IF(present(closed_door)) { 
            IF(present(locked_door)) { 
                IF(not (hasKey())) { 
                    get(key)
                    pickup()
                }  
                get(locked_door)
            }  
            toggle()
        }  
        IF(not (frontIsClear())) { turnRight()}  
        move()
    } ; 
    turnRight()
    WHILE(not (present(goal))) { 
        move()
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
