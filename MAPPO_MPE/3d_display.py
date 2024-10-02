import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv
import time

COLORS = ['red', 'red', 'red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'yellow']
APLHA = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 1.0]

def update_lines(num, dataLines, lines):
    for line, data, color, alpha in zip(lines, dataLines, COLORS, APLHA):
        line.set_data_3d(data[:num,:].T)
        line.set_color(color)
        line.set_alpha(alpha)
    return lines

for num in range(1, 41):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Read data from CSV file
    data = [[] for _ in range(13)]
    num_row = 0
    with open('sim_trajectory/track_{}.csv'.format(num), 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header line
        for row in reader:
            num_row += 1
            for i, dot in enumerate(row):
                pos = [float(dot.strip("[]").split(", ")[j]) for j in range(3)]
                # print(i, pos)
                data[i].append(pos)

    data = np.array(data)
    lines = [ax.plot([],[],[])[0] for _ in data] # create lines intitally without data

    ax.set_title('Trajectory %d' %num)
    ax.set(xlim3d=(-1, 1), xlabel='X')
    ax.set(ylim3d=(-1, 1), ylabel='Y')
    ax.set(zlim3d=(-1, 1), zlabel='Z')
    # Add scatter plot for the constant point
    ax.scatter(data[12][:, 0], data[12][:, 1], data[12][:, 2], c='yellow', alpha=1.0, s=20)

    ani = animation.FuncAnimation(fig, update_lines, num_row, fargs=(data, lines), interval=50)

    plt.show()


