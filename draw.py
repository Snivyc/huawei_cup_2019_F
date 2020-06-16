from myanswer import *

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from public_function import get_info

# print(len(sys.argv))
if len(sys.argv) < 2:
    print("输入需绘制的问题")
    raise RuntimeError


# print(sys.argv[1])

def main(s):
    path = eval(s)

    if s.endswith("1"):
        from config1 import END
    elif s.endswith("2"):
        from config2 import END
    else:
        raise RuntimeError
    path.append(END)

    if s[1] == "1" or s[1] == "3":
        draw_path(path)
    elif s[1] == "2":
        print("画图时r扩大了5倍")
        cycle_draw(path)


def draw_path(PATH):
    plt.rcParams['legend.fontsize'] = 10

    fig = plt.figure(figsize=plt.figaspect(1) * 1.8)
    ax = fig.gca(projection='3d')
    x = []
    y = []
    z = []
    for i in PATH:
        a, b, c = i
        x.append(a)
        y.append(b)
        z.append(c)
    ax.plot(x, y, z, c="black", label='path')

    ax.auto_scale_xyz([0, 100000], [0, 100000], [0, 100000])
    plt.show()


def cycle_draw(path):
    plt.rcParams['legend.fontsize'] = 10
    # plt.figure(figsize=(6, 6.5))

    fig = plt.figure(figsize=plt.figaspect(1) * 1.8)
    ax = fig.gca(projection='3d')

    # ax.auto_scale_xyz([1, 1], [1, 1], [1, 1])

    # ax.set_aspect(1)
    # ax.legend(np.linspace(0, 100000, 10000))

    x = [i[0] for i in path[0:2]]
    y = [i[1] for i in path[0:2]]
    z = [i[2] for i in path[0:2]]
    #  (x)
    ax.plot(x, y, z, c="black", label='path')
    for i in range(len(path) - 2):
        #  (path[i], path[i+1], path[i+2])
        #  (get_cycle_info(path[i], path[i+1], path[i+2]))
        #
        o, d, (x, y, z) = get_info(path[i], path[i + 1], path[i + 2])
        # test_func(path[i], path[i+1], path[i+2])
        # o, d , yuan = get_info(path[i], path[i+1], path[i+2])
        # ax.plot(*yuan, c = "red")
        #  (path[i + 1], d)
        # o = np.array(o)
        # ax.scatter(*o, c = "green")

        #  ("x", x)
        ax.plot(x, y, z, c="red", label='path')

        s = d
        e = path[i + 2]
        x = (s[0], e[0])
        y = (s[1], e[1])
        z = (s[2], e[2])

        #  ("x", x)

        ax.plot(x, y, z, c="black", label='path')
    ax.auto_scale_xyz([0, 100000], [0, 100000], [0, 100000])
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])
