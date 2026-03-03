import json

import matplotlib.pyplot as plt
import numpy as np

def r2(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def get_data():
    with open("cs_data.json", "r") as f:
        raw_data = json.load(f)
        x = []
        y = []
        for key, value in zip(raw_data, raw_data.values()):
            x.append(float(key))
            y.append(value)
    return x, y

def c_s():
    """
    Calculates the speed of sound of stellar matter in a sun-like star as a 
    function of the distance from the star's core, in solar radii.

    :param r: Distance from the center of the star in solar radii.
    :type r: float
    :return: Speed of sound in meters per second.
    :rtype: float
    """
    x, y = get_data()
    x = np.array(list(x))
    y = np.array(list(y))

    coeffs = np.polyfit(x, y, 4)
    p_fit = np.poly1d(coeffs)

    return lambda r: p_fit(r) * 100000 # converting from 100 km/s to 100,000 m/s

def fit(n: int):
    x, y = get_data()
    x = np.array(list(x))
    y = np.array(list(y))

    coeffs = np.polyfit(x, y, n)
    p_fit = np.poly1d(coeffs)

    print(coeffs)

    x_fit = np.linspace(min(x), max(x), 200)
    x_fit_20 = np.linspace(min(x), max(x), 20)
    y_fit = p_fit(x_fit)
    y_fit_20 = p_fit(x_fit_20)
    r_squared = r2(y, y_fit_20)

    plt.scatter(x, y, label="Raw Data")
    plt.plot(x_fit, y_fit, label=fr"Best Fit (deg={n}), $R^2$={r_squared:.4f}")
    plt.xlabel(r"$r$ ($R_\odot$)")
    plt.ylabel(r"$c$ ($100 \cdot km \cdot s^{-1}$)")
    plt.title("Speed of Sound in The Stellar Plasma of a Sun-Like Star as a Function of its Distance From The Star's Core")
    plt.legend()
    plt.show()

def main():
    while True:
        n = input("Enter degree for polynomial best fit: ")
        try:
            n = int(n)
            break
        except:
            print(f"{n} is not a valid number.")
    fit(n)

if __name__ == "__main__":
    main()
