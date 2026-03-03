from typing import Callable
import os
import json
from typing import Tuple
import math

from numpy import e
import scipy

import cs_fit

SOLAR_MASS = 1.989e30
SOLAR_RADIUS = 696340000
sound_speed = cs_fit.c_s()

def I(v: float, c_s: float, theta: float, r: float, e: float, T: float) -> float:
    """
    Approximates the value of I in Ostriker's formula
    for gravitational dynamical friction.

    :param v: Relative velocity of the massive body in
    meters per second.
    :type v: float
    :param c_s: Speed of sound of the surrounding stellar
    mass in meters per second.
    :type c_s: float
    :param theta: True anomaly in radians.
    :type theta: float
    :param r: Radius of the PBH.
    :type r: float
    :param e: Eccentricity of the orbit.
    :type e: float
    :param T: Orbital period in seconds.
    :type T: float
    :return: Approximation of I.
    :rtype: float
    """
    M_s = v / c_s
    if M_s < 1:
        return (M_s ** 3) / 3
    else:
        x = (v * t_q(theta, e, T)) / (r)
        if x == 0:
            return 0
        else:
            return math.log(x)

def mean_motion(T: float) -> float:
    """
    Calculates the mean motion of a body orbiting in an
    elliptical orbit

    :param T: Period of the orbit in seconds
    :type T: int
    :return: Mean motion in radians per second
    :rtype: float
    """
    return (2 * scipy.constants.pi / T)

def tangential_acceleration(F: float, m: float) -> float:
    """
    Calculates the tangential acceleration of the perturbing,
    friction-like force acting on the orbiting body, using
    Newton's Second Law.

    :param F: The magnitude of the force in newtons.
    :type F: float
    :param m: Mass of the object in kilograms.
    :type m: float
    :return: Tangential acceleration in meters per second
    squared.
    :rtype: float
    """
    return F / m

def specific_orbital_energy(M: float, m: float, a: float) -> float:
    """
    Calculates the specific orbital energy of the orbiting body.

    :param M: Mass of the larger body in kilograms.
    :type M: float
    :param m: Mass of the smaller body in kilograms.
    :type m: float
    :param a: Length of the semi-major axis of the elliptical orbit
    in meters.
    :type a: float
    :return: Specific orbital energy in Joules per kilogram.
    :rtype: float
    """
    mu = (M + m) * scipy.constants.G
    return -mu / (2 * a)

def change_in_semi_major_axis(
        T: float, e: float, a: float, m: float
) -> Tuple:
    """
    Approximates the change in the orbit's semi-major axis with a perturbing
    force acting upon the orbiting body.

    :param T: Period of the orbit in seconds.
    :type T: float
    :param a_theta: Tangential acceleration of the orbiting body in meters per
    second squared as a function of time in seconds.
    :type a_theta: float
    :param e: Eccentricity of the orbit's ellipse.
    :type e: float
    :param theta: True anomaly of the orbiting body in radians as a function of 
    time in seconds.
    :param n: Mean motion of the orbiting body in radians per second.
    :type n: float
    :return: Change in the orbit's semi-major axis in meters, and the 
    integration error.
    :rtype: tuple 
    """
    def a_theta(t):
        r = r_from_t(t, a, m, e)
        v = vis_viva_equation(SOLAR_MASS, r, a)
        c_s = sound_speed(r / SOLAR_RADIUS)
        n = mean_motion(T)
        M_theta = mean_anomaly(n, t)
        E, residual = newtons_method_for_E(e, M_theta)
        theta = true_anomaly(E, e)
        I_c = I(v=v, c_s=c_s, theta=theta, r=r, e=e, T=T)
        rho = solar_density_model(r)
        force = F_DF(I=I_c, M=m, rho=rho, v=v)
        acceleration = force / m
        return acceleration
    def theta(t):
        n = mean_motion(T)
        M_theta = mean_anomaly(n, t)
        E, residual = newtons_method_for_E(e, M_theta)
        true_theta = true_anomaly(E, e)
        return true_theta
    n = mean_motion(T)
    num = lambda t: 2 * a_theta(t) * (1 + (e * math.cos(theta(t))))
    denom = n * math.sqrt(1 - (e ** 2))
    integrand = lambda t: num(t) / denom
    result, error = scipy.integrate.quad(integrand, 0, T)
    return result

def change_in_eccentricity(
    T: float, m: float, e: float, a: float,
    epsilon: float
) -> Tuple:
    """
    Approximates the change in the orbit's eccentricity with a perturbing force
    acting upon the orbiting body.

    :param T: Period of the orbit in seconds.
    :type T: float
    :param a_theta: Tangential acceleration of the orbiting body in meters per
    second squared as a function of time in seconds.
    :type a_theta: float
    :param e: Eccentricity of the orbit's ellipse.
    :type e: float
    :param theta: True anomaly of the orbiting body in radians as a function of 
    time in seconds.
    :param a: Semi-major axis of the orbit's ellipse in meters.
    :type a: float
    :param epsilon: Specific orbital energy of the orbiting body in Joules per
    kilogram.
    :type epsilon: float
    :return: Change in the orbit's eccentricity, and the integration error.
    :rtype: tuple 
    """
    def a_theta(t):
        r = r_from_t(t, a, m, e)
        v = vis_viva_equation(SOLAR_MASS, r, a)
        c_s = sound_speed(r / SOLAR_RADIUS)
        n = mean_motion(T)
        M_theta = mean_anomaly(n, t)
        E, residual = newtons_method_for_E(e, M_theta)
        theta = true_anomaly(E, e)
        I_c = I(v=v, c_s=c_s, theta=theta, r=r, e=e, T=T)
        rho = solar_density_model(r)
        force = F_DF(I=I_c, M=m, rho=rho, v=v)
        acceleration = force / m
        return acceleration
    def theta(t):
        n = mean_motion(T)
        M_theta = mean_anomaly(n, t)
        E, residual = newtons_method_for_E(e, M_theta)
        true_theta = true_anomaly(E, e)
        return true_theta
    n = mean_motion(T)
    frac = math.sqrt(1 - (e ** 2)) / (n * a)
    coeff = lambda t: a_theta(t) * (math.cos(theta(t)) + ((e + math.cos(theta(t))) / (1 + e * math.cos(theta(t)))))
    integrand = lambda t: frac * coeff(t)
    result, error = scipy.integrate.quad(integrand, 0, T)
    return result

def orbital_period(a: float, M: float, m: float) -> float:
    """
    Calculates the orbital period of an elliptical orbitusing Kepler's Third 
    Law.

    :param a: Semi-major axis of the orbit in meters.
    :type a: float
    :param M: Mass of the larger body in kilograms.
    :type M: float
    :param m: Mass of the smaller body in kilograms.
    :type m: float
    :return: Orbital period of the orbit in seconds
    :rtype: float
    """
    num = 4 * (scipy.constants.pi ** 2) * (a ** 3)
    denom = scipy.constants.G * (M + m)
    return math.sqrt(num / denom)

def vis_viva_equation(M: float, r: float, a: float) -> float:
    """
    Calculates the relative velocity of the orbiting body in meters per second,
    using the vis-viva_equation.

    :param M: Mass of the central body in kilograms.
    :type M: float
    :param r: Distance of the smaller body from the central body in meters.
    :type r: float:
    """
    inner = (2 / r) - (1 / a)
    return math.sqrt(scipy.constants.G * M * inner)

def solar_density_model(r: float) -> float:
    """
    Model of the sun's density as a function of the distance from its core.

    :param r: Distance from the center of the sun in meters.
    :type r: float
    :return: Approximate density of stellar matter in kilograms per meter cubed.
    :rtype: float
    """
    inner = 1 - (r / SOLAR_RADIUS)
    if inner < 0:
        return 0
    RHO_0 = 1.6e5 # 1.6 * 10^5 kg m^-3
    return RHO_0 * (inner ** 6)

def mean_anomaly(n: float, t: float) -> float:
    """
    Calculates the mean anomaly of the orbit.

    :param n: Mean Motion of the orbiting body.
    :type n: float
    :param t: Time since periapsis in seconds.
    :type t: float
    :return: Mean Anomaly in radians.
    :rtype: float
    """
    return n * t

def eccentricity(a: float, b: float) -> float:
    """
    Calculates the eccentricity of an ellipse given the lengths of its 
    semi-major and semi-minor axes.

    :param a: Length of the ellipse's semi-major axis in meters.
    :type a: float
    :param b: Length of the ellipse's semi-minor axis in meters.
    :type b: float
    :return: Eccentricity of the ellipse.
    :rtype: float
    """
    radicand = 1 - ((b / a) ** 2)
    return math.sqrt(radicand)
    
def newtons_method_for_E(e: float, M_theta: float) -> Tuple:
    """
    Numerically integrates for the Eccentric Anomaly E given the orbit's 
    eccentricity and mean anomaly.

    :param e: Eccentricity of the orbit.
    :type e: float
    :param M_theta: Mean anomaly of the orbit.
    :type M_theta: float
    :return: Approximation of the Eccentric Anomaly and the Residual.
    :rtype: Tuple
    """
    f = lambda E: E - e * math.sin(E) - M_theta
    df = lambda E: 1 - e * math.cos(E)
    try:
        eccentric_anomaly = scipy.optimize.newton(
            f, x0 = M_theta, fprime=df, rtol=1e-12
        )
    except RuntimeError:
        eccentric_anomaly = scipy.optimize.brentq(f, 0, 2 * math.pi)
    residual = f(eccentric_anomaly)
    return eccentric_anomaly, residual

def distance_from_center(a: float, e: float, E: float) -> float:
    """
    Calculates the distance from the center of the star given the Eccentric
    Anomaly.

    :param a: Length of the semi-major axis in meters.
    :type a: float
    :param e: Eccentricity of the orbit.
    :type e: float
    :param E: Eccentric Anomaly in radians.
    :type E: float
    :return: Distance from the center of the star in meters.
    :rtype: float
    """
    return a * (1 - e * math.cos(E))

def true_anomaly(E: float, e: float) -> float:
    """
    Calculates the true anomaly given the eccentric anomaly.

    :param E: Eccentric anomaly in radians.
    :type E: float
    :param e: Eccentricity of the orbit.
    :type e: float
    :return: True anomaly in radians.
    :rtype: float
    """
    num_cos = math.cos(E) - e
    denom_cos = 1 - (e * math.cos(E))
    cos_theta = num_cos / denom_cos

    num_sin = math.sqrt(1 - (e ** 2)) * math.sin(E)
    denom_sin = 1 - (e * math.cos(E))
    sin_theta = num_sin / denom_sin

    return math.atan2(sin_theta, cos_theta)

def change_in_mass(
    alpha: float, m: float, T: float, a: float, e: float
) -> float:
    """
    Numerically integrates the change in mass of the star and the PBH due to
    Bondi-Hoyle-Lyttleton accretion.

    :param alpha: Dimensionless factor between 1 and 2. Determined empirically.
    :type alpha: float
    :param m: Mass of the black hole in kilograms.
    :type m: float
    :param T: Orbital period in seconds.
    :type T: float
    :param a: Semi-major axis of the orbit in meters.
    :type T: float
    :param e: Eccentricity of the orbit.
    :type T: float
    :return: Change in mass of the black hole in kilograms.
    :rtype: float
    """
    r = lambda t: r_from_t(t, a, m, e)
    v = lambda t: vis_viva_equation(SOLAR_MASS, r(t), a)
    num = lambda t: alpha * 2 * math.pi * ((scipy.constants.G * m) ** 2) * solar_density_model(r(t))
    denom = lambda t: math.sqrt(((v(t) ** 2) + (sound_speed(r(t) / SOLAR_RADIUS)) ** 2) ** (3))
    integrand = lambda t: num(t) / denom(t)
    result, error = scipy.integrate.quad(integrand, 0, T)
    return result

def schwartzchild_radius(m: float) -> float:
    """
    Calculates the Schwartzchild Radius of an object of mass m.

    :param m: Mass of the object in kilograms.
    :type m: float
    :return: Schwartzchild radius of the object in meters.
    :rtype: float
    """
    return (2 * scipy.constants.G * m) / (scipy.constants.c ** 2)

def perifocal_distance(a: float, e: float) -> float:
    """
    Calculates the perifocal distance of the ellipse.

    :param a: Length of the semi-major axis in meters.
    :type a: float
    :param e: Eccentricity of the ellipse.
    :type e: float
    :return: Perifocal distance of the ellipse in meters.
    :rtype: float
    """
    return a * (1 - e)

def t_from_theta(theta: float, e: float, T: float) -> float:
    """
    Calculates the time it would take from periapsis for the orbiting body to 
    reach the true anomaly theta.

    :param theta: True anomaly in radians.
    :type theta: float
    :param e: Eccentricity of the orbit.
    :type e: float
    :param T: Orbital period in seconds.
    :type T: float
    :return: Time in seconds.
    :rtype: float
    """
    cos_E = (e + math.cos(theta)) / (1 + (e * math.cos(theta)))
    sin_E = (math.sin(theta) * math.sqrt(1 - (e ** 2))) / (1 + (e * math.cos(theta)))
    E = math.atan2(sin_E, cos_E)
    M_theta = E - (e * math.sin(E))
    n = mean_motion(T)
    return M_theta / n

def t_q(theta: float, e: float, T: float) -> float:
    """
    Calculates t_q from the true anomaly.

    :param theta: True anomaly in radians.
    :type theta: float
    :param e: Eccentricity of the orbit.
    :type e: float
    :param T: Orbital period in seconds.
    :type T: float
    :return: Time in seconds.
    :rtype: float
    """
    return T / 4

def r_from_t(t, a, m, e):
    T = orbital_period(a, SOLAR_MASS, m) 
    n = mean_motion(T)
    M_theta = mean_anomaly(n, t)
    E, residual = newtons_method_for_E(e, M_theta)
    r = distance_from_center(a, e, E)
    return r

def F_DF(I: float, M: float, rho: float, v: float) -> float:
    return (-I) * 4 * math.pi * ((scipy.constants.G * M) ** 2) * (rho / (v ** 2))

def approx_decay(initial_a, initial_e, initial_m, sun_mass, orbits):
    pbh_mass = initial_m
    semi_major = initial_a
    ecc = initial_e
    T = orbital_period(semi_major, sun_mass, pbh_mass)
    epsilon = specific_orbital_energy(sun_mass, pbh_mass, semi_major)
    ecc += orbits * change_in_eccentricity(T, pbh_mass, ecc, semi_major, epsilon)
    semi_major += orbits * change_in_semi_major_axis(T, initial_e, initial_a, pbh_mass)
    return semi_major, ecc

def main(pbh_mass=None, orbits_per_batch=None):
    if orbits_per_batch == None:
        orbits_per_batch = int(input("How many orbits per batch?: "))

    a = SOLAR_RADIUS
    e = 0.1

    if pbh_mass == None:
        pbh_mass = 1e29
    sun_mass = SOLAR_MASS
    time = 0.0

    a_s = [a]
    e_s = [e]
    times = [time]
    pbh_masses = [pbh_mass]
    sun_masses = [sun_mass]
    distances = [distance_from_center(a, e, 0)]

    print(f"Starting a: {a}, e: {e}")

    CORE_RADIUS = SOLAR_RADIUS / 5
    
    i = 0
    while True:
        i += 1
        print(f"Orbit #{(i * orbits_per_batch)}")
        T = orbital_period(a, sun_mass, pbh_mass)
        time += orbits_per_batch * T
        delta_m = orbits_per_batch * change_in_mass(1, pbh_mass, T, a, e)
        da, de = approx_decay(a, e, pbh_mass, sun_mass, orbits_per_batch)
        a = max(0, da)
        e = max(0, de)
        a_s.append(a)
        e_s.append(e)
        pbh_mass += delta_m
        sun_mass -= delta_m
        pbh_masses.append(pbh_mass)
        sun_masses.append(sun_mass)
        times.append(time)
        distance = distance_from_center(a, e, 0)
        distances.append(distance)
        print(f"a: {a} e: {e}, M: {sun_mass}, m: {pbh_mass}, time: {time / 3.154e7} years")
        if distance < CORE_RADIUS:
            print(f"Within {CORE_RADIUS} meters of the center of the sun!")
            break
    print("Done!")
    with open(f"results/initial_mass_1e+{int(math.log10(pbh_masses[0]))}_batch_size_{orbits_per_batch}.json", "w") as f:
        data = {
            "time": times,
            "semi-major axis": a_s,
            "eccentricity": e_s,
            "pbh_mass": pbh_masses,
            "sun_mass": sun_masses,
            "distances": distances
        }
        json.dump(data, f, indent=4)

from scipy.integrate import solve_ivp

def orbital_decay_system(t, y):
    """
    ODE system:
        y = [a, e, m, M]
    """

    a, e, m, M = y

    # Stop invalid states
    if a <= 0 or e < 0 or M <= 0:
        return [0.0, 0.0, 0.0, 0.0]

    # Orbital period using evolving M
    T = orbital_period(a, M, m)

    # Specific orbital energy must also use evolving M
    energy = specific_orbital_energy(M, m, a)

    # Orbit-averaged changes per orbit
    delta_a = change_in_semi_major_axis(T, e, a, m)
    delta_e = change_in_eccentricity(T, m, e, a, energy)
    delta_m = change_in_mass(1, m, T, a, e)

    # Convert to time derivatives
    da_dt = delta_a / T
    de_dt = delta_e / T
    dm_dt = delta_m / T

    # Mass conservation
    dM_dt = -dm_dt

    return [da_dt, de_dt, dm_dt, dM_dt]

def core_reached_event(t, y):
    a, e, m, M = y
    r_peri = distance_from_center(a, e, 0)
    return r_peri - (SOLAR_RADIUS * 0.2)

core_reached_event.terminal = True
core_reached_event.direction = -1

def run_decay_simulation(initial_a, initial_e, initial_m,
                         max_time_years=1e13):

    y0 = [initial_a, initial_e, initial_m, SOLAR_MASS]

    t_span = (0, max_time_years * 3.154e7)

    solution = solve_ivp(
        orbital_decay_system,
        t_span,
        y0,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
        events=core_reached_event
    )

    return solution

def rk(initial_a, initial_e, initial_m):
    initial_sun_mass = SOLAR_MASS

    sol = run_decay_simulation(initial_a, initial_e, initial_m)

    final_time = sol.t[-1]
    final_a = sol.y[0][-1]
    final_e = sol.y[1][-1]
    final_pbh_mass = sol.y[2][-1]
    final_sun_mass = sol.y[3][-1]

    print("Simulation finished.\n")

    print("Final time (years):", final_time / 3.154e7)
    print("Final semi-major axis a:", final_a)
    print("Final eccentricity e:", final_e)
    print("Final PBH mass:", final_pbh_mass)
    print("Final Sun mass:", final_sun_mass)

    with open(f"results/{(initial_a / SOLAR_RADIUS):.2f}_{initial_e}/{initial_m}.json", "w") as f:
        data = {
            "initial a": initial_a,
            "final a": final_a,
            "initial e": initial_e,
            "final e": final_e,
            "initial m": initial_m,
            "final m": final_pbh_mass,
            "initial M": initial_sun_mass,
            "final M": final_sun_mass,
            "time": final_time / 3.154e7
        }
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    cases = [
        [SOLAR_RADIUS, 0.0167],
        [SOLAR_RADIUS, 0.5],
        [SOLAR_RADIUS, 0.9],
        [SOLAR_RADIUS / 2, 0.0167],
        #[SOLAR_RADIUS / 2, 0.5],
        [SOLAR_RADIUS / 2, 0.9],
        [SOLAR_RADIUS * 3 / 4, 0.0167],
        [SOLAR_RADIUS * 3 / 4, 0.5],
        [SOLAR_RADIUS * 3 / 4, 0.9],
        [SOLAR_RADIUS * 3, 0.9],
        [SOLAR_RADIUS * 5, 0.95]
    ]

    masses = [1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29]

    import subprocess
    total = len(cases) * len(masses)
    i = 0
    for case in cases:
        for mass in masses:
            i += 1
            if not os.path.exists(f"results/{(case[0] / SOLAR_RADIUS):.2f}_{case[1]}"):
                subprocess.run(f"mkdir results/{(case[0] / SOLAR_RADIUS):.2f}_{case[1]}".split())
            if not os.path.exists(f"results/{(case[0] / SOLAR_RADIUS):.2f}_{case[1]}/{mass}.json"):
                rk(case[0], case[1], mass)
                print(f"{i}/{total}")
