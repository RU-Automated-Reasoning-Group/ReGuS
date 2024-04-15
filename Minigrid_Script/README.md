# MiniGrid

To run the code for MiniGrid environments, use conda env regus 2 by running
```
conda activate regus2
```

Install MiniGrid Package
```
cd Minigrid
python3 -m pip install -e .
```

## 1.  Synthesizing a new program for a specific minigrid environment with library

ReGuS is capable of complicated programs based simple programs that synthesized for other environments, i.e. curriculum synthesis. For each environment, represented as an index number, we provide a checkpoint library that contains all synthesized programs for previous environments. Such a library will be used as building blocks for later environments.

```
# for index 0, use
python3 sequence.py --run_one


# for index > 0, use the following command
# substitute [IDX] with the actually number

python3 sequence.py --idx [IDX] --library checkpoints/library_ckpt_[IDX] --library_dict checkpoints/library_dict_ckpt_[IDX] --run_one
```

Different index represents a different MiniGrid environment, shown in the following table

| Index      | Environment                            | Estimated Time    |
| -----------| ---------------------------------------|-------------------|
| 0          | MiniGrid-RandomCrossingS11N5-v0        | 30 mins           |
| 1          | MiniGrid-RandomLavaCrossingS11N5-v0    | 2 mins            |
| 2          | MiniGrid-MultiRoomNoDoor-N6-v0         | 2 mins            |
| 3          | MiniGrid-MultiRoom-N6-v0               | 2 mins            |
| 4          | MiniGrid-LockedRoom-v0                 | 40 mins           |
| 5          | MiniGrid-DoorKey-8x8-v0                | 30 mins           |
| 6          | MiniGrid-PutNearTwoRoom-v0             | 20 mins           |
| 7          | MiniGrid-UnlockPickup-v0               | 15 mins           |

Once the synthesis terminates for the selected in, a text file that has the same name as the environment will be generated in the `results` folder. This text file contains the synthesized program. The output of this program is a detailed log describing the synthesis procedure. However, the output can be very long so output should be discard to `/dev/null` or splited into multiple files such as 

```
python3 sequence.py --idx 1 --library checkpoints/library_ckpt_1 --library_dict checkpoints/library_dict_ckpt_1 --run_one | split -d -b 50m - log.
```

## 2. Full Curriculum Synthesis
Use the following command 
```
python3 sequence.py | split -d -b 50m - log.
```

This is equivalent to running each of the seperated environments in one run. The program will take about 2.5 hours to finish and will produce a synthesized program for each environment. 

The steps used to synthesize the programs can be found in the logs files at the end of searching for each environment. Searching "Moving to next env" to check the number of steps used. An figure called `minigrid.png` will be generated to show the number of environments steps needed to solve the environments.


### Code Structure
- `checkpoints` contains the provided library checkpoint for each environment
- `Minigrid` contains the MiniGrid environment
- `mcts` folder and `mcts_search.py`contain the implementation of the Monte Carlo Tree Search (MCTS) used by ReGus
- `results` folder contains the generated program
- `minigrid_base_dsl.py` specify the action and predicate sets used for each of the environments. This file also implements the Domain Specific Language (DSL) used by ReGuS. This file should be considered as the interface of ReGuS that users will most probabily change for new environents or test new actions. The actually implementation of actions and predicates are defined in the two following locations.
- `Minigrid/minigrid/minigrid_env.py`: This file is contains the base class for all minigrid environments and also contains the implementation of all predicates. For example, the `front_is_clear` predicate is defined in the class method also called `front_is_clear`
- `minigrid_implement` folder contains the actual implementation of the low level actions (what is underhood of actions from a ReGuS program) that directly interact with the environment
- `search.py` and `search2.py` contains the partial program executor for ReGuS DSL and implement the main logic of ReGuS, such as generate new actions within a program or synthesize if/ifelse when necessary. `search2.py` always use multiple seeds to evaluate the performance of a program
- `sequence.py` contains the curriculum synthesis procedure that synthesize program for a new environments using existing programs from previous environments. In this file, we also set hyperparameters for ReGuS for different environment. We also uses some heuristics that the synthesized programs must satisfy to make them more likely to generalize in later environments. ReGuS will possibly generate multiple programs that works for the current environments, however these programs are not equally generalizable for more complex environments.
 