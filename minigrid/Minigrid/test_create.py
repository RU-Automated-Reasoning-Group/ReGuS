import minigrid
import gymnasium as gym
import pdb

minigrid.register_minigrid_envs()
env_name = "MiniGrid-RandomCrossingR2LS11N5-v0"
a = gym.make(env_name, render_mode="rgb_array") 
for i in range(0, 10):
    a.reset()
    # pdb.set_trace()
    a.env.env.render(dir=f"test_frames/img{i:03d}.png")
    obs, rwd, _, _, _ = a.step(1)
    print(obs)
    a.env.env.render(dir=f"test_frames/img_act{i:03d}.png")
