import matplotlib.pyplot as plt

def plot_initial_positions(file_name, out_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        xs = []
        ys = []
        for line in lines:
            line = line.split()
            ns = [float(x) for x in line]
            xs.append(ns[0])
            ys.append(ns[1])
    
    plt.scatter(xs, ys)
    plt.savefig(out_name)

if __name__ == "__main__":
    plot_initial_positions("original.txt", "original.png")
    plot_initial_positions("optimized.txt", "optimized.png")

