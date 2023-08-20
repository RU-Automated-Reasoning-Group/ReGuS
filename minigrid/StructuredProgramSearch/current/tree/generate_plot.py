from matplotlib import pyplot as plt
import numpy as np


def process_all_data(all_data):
    mean_counter = 0.0
    means = []
    stds = []
    for data in all_data:
        mean_data = np.mean(data + mean_counter)
        std_data = np.std(data)

        mean_counter = mean_data

        means.append(mean_data)
        stds.append(std_data)

    return np.array(means), np.array(stds)


def process_all_data_prof(all_data):
    n_passes = len(all_data[0])
    passes = [[x[i] for x in all_data] for i in range(0, n_passes)]

    for i in range(0, n_passes):
        count = passes[i][0]
        for j in range(1, len(passes[i])):
            count += passes[i][j]
            passes[i][j] = count

    print(passes)
    max_interactions = max([p[-1] for p in passes])
    xs = []
    ys = []
    stds = []
    print(max_interactions)
    # exit()
    for i in range(0, max_interactions + 100000):
        if i % 10000 == 0 or i >= max_interactions - 10000:
            # if True:
            print(i)
            solved = []
            for p in passes:
                p_solve = 0
                for idx in range(0, len(p)):
                    if i > p[idx]:
                        p_solve = idx + 1
                solved.append(p_solve)

            xs.append(i)
            ys.append(np.mean(solved))
            stds.append(np.std(solved))
    return np.array(xs), np.array(ys), np.array(stds), max_interactions


def envs_plot(output_name="minigrid_figure.png", myplot=False):
    # random crossing
    random_crossing_data = np.array([433486, 628968, 1016276])
    random_crossing_data_baseline = np.array([465005, 1017965, 720018])

    # lava crossing
    lava_crossing_data = np.array([2492, 1373, 1945])
    lava_crossing_data_baseline = 2 * np.array([465005, 1017965, 720018])

    # multiroom
    multiroom_nodoor_data = np.array([8093, 5831, 20109])
    multiroom_data = np.array([17266, 15021, 14776])

    multiroom_data = multiroom_data + multiroom_nodoor_data

    # lockedroom
    lockedroom_data = np.array([353304, 357463, 410800])

    # doorkey data
    doorkey_data = np.array([7515470, 666001, 4458032])

    # unlockedpickup data
    unlocked_pickup = np.array([77143, 101776, 65483])

    # all data
    all_data = [
        random_crossing_data,
        lava_crossing_data,
        multiroom_data,
        lockedroom_data,
        doorkey_data,
        unlocked_pickup,
    ]

    all_baseline_data = [random_crossing_data_baseline, lava_crossing_data_baseline]

    if myplot:
        values, stds = process_all_data(all_data)

        x = np.arange(0, len(values))
        plt.plot(x, values, "-", color="blue")
        # plt.fill_between(x, values + stds, values - stds, alpha=0.5, facecolor='#FF9848')
        plt.errorbar(x, values, stds)

        # xticks
        labels = [
            "RandomCrossing",
            "RandomLavaCrossing",
            "MultiRoom",
            "LockedRoom",
            "DoorKey",
            "UnlockedPickup",
        ]
        plt.xticks(x, labels, rotation=-12)

        # x, y labels and title
        plt.xlabel("Environment name")
        plt.ylabel("Number of interactions")
        plt.title("Title")

        plt.savefig(output_name)
    else:
        xs, ys, stds, max_iters = process_all_data_prof(all_data)
        bxs, bys, bstds, _ = process_all_data_prof(all_baseline_data)
        plt.plot(xs, ys, "-", color="blue", label="with curriculum synthesis")
        plt.plot(bxs, bys, "-", color="red", label="w/o curriculum synthesis")
        plt.fill_between(xs, ys + stds, ys - stds, alpha=0.5, facecolor="cyan")
        plt.fill_between(bxs, bys + bstds, bys - bstds, alpha=0.5, facecolor="pink")
        plt.plot([bxs[-1], max_iters], [bys[-1], bys[-1]], "-", color="red")
        plt.xlabel("Number of Environment Interactions")
        plt.ylabel("Task Solved")
        plt.title("MiniGrid Tasks")
        plt.legend(loc="lower right")
        plt.savefig(output_name)


if __name__ == "__main__":
    import sys

    num = int(sys.argv[1])
    if num == 1:
        envs_plot("fig1.png", myplot=False)
        print("fig1 saved")
    elif num == 2:
        envs_plot("fig2.png", myplot=True)
        print("fig2 saved")
    else:
        assert False
