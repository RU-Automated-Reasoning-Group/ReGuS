# How to run the the search

1. Clone this repository. Also clone the [programskill](https://github.com/RU-Automated-Reasoning-Group/programskill) repo.
2. Install the reskill package: In the root directory of the programskill repo, run
```
pip3 install -e .
```
3. Run the search: Go back to the PBR repo
```
cd current/tree
python3 search.py
```

# How to change reward schema

## Change the reward of the actions

In programskill/dsl.py, line 165 - 170 is used to change the reward of each action. The `rwd` varaible returned by the `k.step` function only 1 when the block reaches the goal position.

We can modify this reward. We can also use `k.[predicate_name]()` to check the each predicate if necessary.

### Make video for visiualization
Also set DSL_DEBUG variable in this file to true (currently false) will generate useful debug information, including generating images after each action but excluding idle.

Then run the make_video.sh will generate a video called test_single_sim.mp4 by stacking images in frames/ directory. Remember to remove all images in frames/ directory before next search to make a new video. The images will not be delted automatically.

Setting DSL_DEBUG to true will slow down the search process because lots of images are generated.


## Change how the reward is accumulated
In programskill/robot.py, in function execute_single_action, we have 

```
tmp_rwd = action(self.env, self)
```

tmp_rwd is the reward returned by each action (described in previous section). The self.reward is the cumulative reward. So we can define different reward accumulation strategy here.