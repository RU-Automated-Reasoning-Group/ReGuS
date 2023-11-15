import minigrid
import gymnasium as gym
import pdb
import policies

minigrid.register_minigrid_envs()
env_name = "MiniGrid-LockedRoomR2L-v0"
a = gym.make(env_name, render_mode="rgb_array") 
for i in range(0, 10):
    a.reset()
    # pdb.set_trace()
    a.env.env.render(dir=f"test_frames/img{i:03d}.png")
    # pdb.set_trace()
    obs, rwd, _, _, info = a.step(1)
    print(obs)
    print(info)
    a.env.env.render(dir=f"test_frames/img_act{i:03d}.png")
