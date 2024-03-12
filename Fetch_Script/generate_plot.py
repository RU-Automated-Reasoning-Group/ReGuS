import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pdb



def generate_plot(bench):
    path = "./"
    if bench == "pickandplace":
        prefix = "pick"
        output_file = "figure_pick_reward.pdf"
        title = "Fetch-Pick&Place"
    elif bench == "hook":
        prefix = "hook"
        output_file = "figure_hook_reward.pdf"
        title = "Fetch-Hook"
    else:
        assert False



    path_files = os.listdir(path)

    plt.cla()
    plt.figure()


    for f in path_files:
        if f.startswith(prefix) and f.endswith("csv"):
            print(f)
            df = pd.read_csv(os.path.join(path, f))
            time_steps = np.array(df['Step'])
            value = np.array(df['Value'])
            plt.plot(time_steps, value)


    plt.title(title)
    plt.xlabel("timesteps")
    plt.ylabel("rewards")
    plt.savefig(output_file)

if __name__ == '__main__':
    generate_plot("pickandplace")
    generate_plot("hook")