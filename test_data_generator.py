import numpy as np

if __name__ == '__main__':
    points = 10

    x = np.linspace(0, np.pi, points)
    y = [np.cos(x_) for x_ in x]

    for x_, y_ in zip(x, y):
        print(f"{x_}, {y_}")
