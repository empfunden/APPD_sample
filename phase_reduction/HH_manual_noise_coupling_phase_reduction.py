import sys
import os
import subprocess
import math
import time
from datetime import date, datetime
import numpy as np
import random
import scipy.io
import scipy.integrate
from scipy.stats import multivariate_normal, norm
from scipy.linalg import qr
from scipy.special import erf
from scipy.interpolate import lagrange
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import glob
# import ipdb
import copy

DIM = 4
global T
global limit_cycle
global voltage_record
global coupling_strength_sum
particle_list = []
coupling_strength_sum = 0.0

class Particle:
    # Random initialization
    def __init__(self, weight) -> None:
        self.weight = weight
        self.mu = np.random.rand(DIM)
        M = np.random.rand(DIM, DIM)
        Q, R = qr(M)
        # "Shrink" to avoid being outside domain
        self.M = 0.01 * Q.dot(Q.T)


def particle_to_G(p):
    G = np.concatenate((p.mu.reshape(1,DIM).ravel(), p.M.ravel()), axis=0)
    return G

start_time = time.time()


def restrict_helper(X: np.ndarray) -> bool:
    """
    This is a helper function for restrict_to_domain.
    Given a point, restrict all dimensions (except for voltage) to [0.0, 1.0].
    The returned boolean indicates whether the point needed to be modified.
    Note that this function directly modifies the point X.
    """
    was_modified = False

    for j in range(1, DIM):
        if X[j] < 0.0:
            X[j] = 0.0
            was_modified = True
        elif X[j] > 1.0:
            X[j] = 1.0
            was_modified = True

    return was_modified


def restrict_to_domain(p: Particle) -> None:
    """
    Given a particle, restrict its center and level set points to a domain.
    If the center is outside, we can easily move it back inside.
    If a level set point is outside, we can "flip" it across the center;
    this should almost always suffice to move it into the domain.
    """
    # old_mu = np.copy(p.mu)
    is_outside = restrict_helper(p.mu)
    # if is_outside:
        # print("Moved mu:", old_mu, p.mu)
    
    # print("Regardless, mu is now at", p.mu, "&HHnorm:", np.linalg.norm(HH_advection_velocity(p.mu)))

    for j in range(DIM):
        level_set_point = p.mu + p.M[:, j]
        # Note that level_set_point, a temp variable, may be modified.
        is_outside = restrict_helper(level_set_point)
        if is_outside:
            p.M[:, j] = -p.M[:, j]
            # print("Flipped point:", p.mu - p.M[:, j], "\nrestricted to", level_set_point, "\nnow at", p.mu + p.M[:, j])

# Hodgkin-Huxley parameters.
gna = 120
ena = 115
gk = 36.0
ek = -12
gl = 0.3
el = 10.613
appcur = 10.0

# Update 2024/09/09: these are fake dummy values. 
# The real values should be given in the first 2 command line arguments.
diff_coeff = 1e9
coupling_coeff = 1e9

def HH_advection_velocity(X): # four-dimensional
    dXdt = np.empty([4])

    V = 100.0 * X[0]
    V += 1e-6 * int(
        abs(V - 10.0) < 5e-7 or abs(V - 25.0) < 5e-7 or abs(V - 50.0) < 5e-7
    )  # don't want V = 10 or 25 or 50
    M, N, H = X[1], X[2], X[3]

    try:
        Ah = 0.07 * math.exp(-V / 20.0)
        Am = (25.0 - V) / (10.0 * (math.exp((25.0 - V) / 10.0) - 1.0))
        An = (10.0 - V) / (100.0 * (math.exp((10.0 - V) / 10.0) - 1.0))
        Bh = 1.0 / (math.exp((30.0 - V) / 10.0) + 1.0)
        Bm = 4.0 * math.exp(-V / 18.0)
        Bn = 0.125 * math.exp(-V / 80.0)

        # 0.01 * dV/dt, dm/dt, dn/dt, dh/dt, in that order.
        dXdt[0] = 0.01 * (appcur + gna * (M**3) * H * (ena - V) + gk * (N**4) * (ek - V) + gl * (el - V))
        dXdt[1] = Am * (1 - M) - Bm * M
        dXdt[2] = An * (1 - N) - Bn * N
        dXdt[3] = Ah * (1 - H) - Bh * H
    except OverflowError:
        print("Overflow error. Location is ", X)

    return dXdt

# Reformulation of HH with advection, coupling, noise velocities. 
# Works on G, which is the center concatenated with sqrt of cov matrix.
def HH_advection_velocity_G(t, G):
    global coupling_strength_sum
    dMdt = np.empty([DIM,DIM])
    dmudt = np.empty([DIM])
    mu = G[:DIM]
    M = G[DIM:].reshape(DIM,DIM)
    # Contribution from noise (see eqn. 10 in APPD paper)
    # This (directly taking inverses) differs from the Tikhonov-Cholesky thing in the C++ code.
    dMdt = diff_coeff * np.linalg.inv(M.T)

    # Temporary test: We're interested in how much coupling contributes (barely).
    # v1 = HH_advection_velocity(mu)
    # v2 = coupling_velocity(mu, coupling_strength_sum)
    # fraction = np.linalg.norm(v2) / np.linalg.norm(v1)
    # if fraction > 0.0001:
    #     print("Fraction contributed by coupling: ", fraction)

    # global recentest_weight
    # if abs(recentest_weight - 0.0004387247371484626) < 1e-8:
    #     print("hello; mu=", mu, "\n&veloces:", HH_advection_velocity(mu), coupling_velocity(mu, coupling_strength_sum))

    # Contribution from advection and coupling velocities.
    dmudt = 0
    for j in range(DIM):
        point1 = mu + M[:,j]
        point2 = mu - M[:,j]
        adv_coup_velocity = (HH_advection_velocity(point1) + coupling_velocity(point1, coupling_strength_sum) - HH_advection_velocity(point2) - coupling_velocity(point2, coupling_strength_sum)) / 2
        dMdt[:,j] += adv_coup_velocity
        dmudt += (HH_advection_velocity(point1) + coupling_velocity(point1, coupling_strength_sum) + HH_advection_velocity(point2) + coupling_velocity(point2, coupling_strength_sum)) / 2
    dmudt = dmudt/DIM
    dGdt = np.concatenate((dmudt.reshape(1,DIM).ravel(), dMdt.ravel()), axis=0)
    return dGdt

def HH_coupling_contribution(weight, V, V_prev, sigma_v, sigma_v_prev, t):
    # Zero contribution when voltage is decreasing.
    if V <= V_prev:
        return 0

    threshold_voltage = 0.45
    V_normalized = (V - threshold_voltage) / sigma_v
    V_prev_normalized = (V_prev - threshold_voltage) / sigma_v_prev
    if V_normalized <= V_prev_normalized:
        return 0
    population_proportion = 0.5 * (erf(V_normalized / math.sqrt(2)) - erf(V_prev_normalized / math.sqrt(2)))

    # Zero contribution when voltage is decreasing.
    if V_normalized <= V_prev_normalized:
        return 0
    
    return weight * population_proportion * coupling_coeff / t


def coupling_velocity(x, coupling_strength_sum):
    dxdt = np.zeros(DIM)
    # Coupling potential is 35 mV, rescaled to 0.35.
    dxdt[0] = coupling_strength_sum * (0.35 - x[0])
    return dxdt


def lin_approx_rel_error(center, offset, advection_eqn):
    center_deriv = advection_eqn(center)
    # Avoid checking if center derivative is very small.
    center_norm = np.linalg.norm(center_deriv)
    if (center_norm < 1e-9):
        return 0.0
    near = center + offset
    far = center + 2 * offset
    near_deriv = advection_eqn(near) - center_deriv
    far_deriv = 0.5 * (advection_eqn(far) - center_deriv)
    rel_error = np.linalg.norm(near_deriv - far_deriv) / center_norm
    # print("Error = ", rel_error)
    return rel_error

def particle_error(p, advection_eqn):
    U, S, _ = np.linalg.svd(np.matmul(p.M, p.M.T))
    U_col_1 = U[:, 0]
    offset_1 = U_col_1 * math.sqrt(S[0])
    err_1 = lin_approx_rel_error(p.mu, offset_1, advection_eqn)
    U_col_2 = U[:, 1]
    offset_2 = U_col_2 * math.sqrt(S[1])
    err_2 = lin_approx_rel_error(p.mu, offset_2, advection_eqn)
    if err_1 > err_2:
        return offset_1, err_1
    else:
        return offset_2, err_2


# update_particle_at_index_single_step
def upaiss(timestep: float, p: Particle):
    G = particle_to_G(p)
    p_new = Particle(p.weight)
    # Track a lower timestep (where we know error is small) and upper timestep 
    # (where error might be too large). Our goal is to find a "sweet spot"
    # where the error is within 0.8 of the tolerance.
    tolerance = 0.05
    # lowerstep = 0.0
    # upperstep = timestep
    needsplit = True

    # minadjustment = min(0.005, timestep) # If attempted timestep is < 0.005, then just integrate once
    minadjustment = 0.005
    adjustment = timestep
    currentstep = timestep
    time_before_split = 0.0
    did_once = False # Ensures the loop goes through at least once

    # print("Weight&timestep:", p.weight, timestep)
    # global recentest_weight
    # recentest_weight = p.weight
    # restrict_to_domain(p)

    while (not did_once) or (adjustment > minadjustment):
        did_once = True
        sol = scipy.integrate.solve_ivp(
            HH_advection_velocity_G,
            [0, currentstep],
            G,
            method='RK45',
            t_eval=[currentstep],
            first_step=currentstep,
            max_step=currentstep
        )
        # Convert back from integrated G to updated particle
        p_new.mu = sol.y[:DIM].reshape(DIM)
        p_new.M = sol.y[DIM:].reshape(DIM,DIM)
        offset, error = particle_error(p_new, HH_advection_velocity)

        # print("Error = ", error)

        if currentstep == timestep and error < tolerance: # Good on first try
            time_before_split = currentstep
            needsplit = False
            break
        elif error > tolerance: # Error too large, try smaller timestep
            adjustment *= 0.5
            currentstep -= adjustment
        elif error > 0.8 * tolerance: # Split if within 20% of error bound
            time_before_split = currentstep
            needsplit = True
            break
        else:
            # time_before_split = currentstep
            adjustment *= 0.5 # Try bigger timestep
            currentstep += adjustment
    
    if time_before_split == 0.0: # Which is to say, we kept shrinking the timestep (case 2)
        needsplit = True
        time_before_split = currentstep

  
    
    # print("Time before split:", time_before_split)
    if time_before_split == 0.0:
        print("Error")
        print(timestep, "is timestep")
    sol = scipy.integrate.solve_ivp(
        HH_advection_velocity_G,
        [0, time_before_split],
        G,
        method='RK45',
        t_eval=[time_before_split],
        first_step=time_before_split,
        max_step=time_before_split
    )

    # Trying to visualize where domain escaping occurs.
    # new_location = np.copy(sol.y[:DIM].reshape(DIM))
    # if restrict_helper(np.copy(new_location)):
    #     print("Out of domain at", new_location, ", old at", p.mu)

    p.mu = sol.y[:DIM].reshape(DIM)
    p.M = sol.y[DIM:].reshape(DIM,DIM)
    offset, error = particle_error(p, HH_advection_velocity)

    # We have directly modified p (which is passed by reference),
    # and return information about whether we need to split (and the split direction, if so).
    # print("Returning::::", needsplit, timestep - time_before_split)
    return needsplit, timestep - time_before_split, offset


# update_particle_at_index
def upai(timestep, coupling_timestep, p):

    if timestep == 0.0:
        return 0.0
    
    V_prev = p.mu[0]
    sigma_prev = p.M[0, 0]

    needsplit, remaining_time, split_dir = upaiss(timestep, p)

    new_split_dir = particle_error(p, HH_advection_velocity)

    # Case 1: Integrated entire timestep without needing to split.
    if not needsplit:
        return HH_coupling_contribution(p.weight, p.mu[0], V_prev, p.M[0, 0], sigma_prev, coupling_timestep)
    
    # Case 2: Made it partially through; need to split (possibly recursively).
    else:
        coupling_now = HH_coupling_contribution(p.weight, p.mu[0], V_prev, p.M[0, 0], sigma_prev, coupling_timestep)
        # Todo: low weight relax treatment.
    
        # Low weight relax treatment: shrink instead of split
        if (p.weight < 0.0005):
            # print("Low weight relaxing")
            p.M /= 2.0
            return coupling_now + upai(remaining_time, coupling_timestep, p)
        c, l, r = split_particle_in_direction(p, split_dir)

        # New idea: don't bother splitting if daughter particles are very close (within combine radius)
        global KD
        distance_A, index_A = KD.query(c.mu)
        phase_A = 2.0 * np.pi * index_A / len(limit_cycle)
        distance_B, index_B = KD.query(l.mu)
        phase_B = 2.0 * np.pi * index_B / len(limit_cycle)
        if abs(phase_A - phase_B) < 0.05:
            print(f"Low dist relaxing: {np.linalg.norm(c.mu-l.mu):.03f}, {p.weight:.03f}")
            p.M /= 2.0
            return coupling_now + upai(remaining_time, coupling_timestep, p)

        p.weight = c.weight
        p.mu = c.mu
        p.M = c.M
        # print("Splitted weight:", c.weight)
        particle_list.append(l)
        particle_list.append(r)
        # ipdb.set_trace
        return coupling_now + upai(remaining_time, coupling_timestep, p) + \
            upai(remaining_time, coupling_timestep, l) + \
            upai(remaining_time, coupling_timestep, r)
    

def update_ODE_adaptive_split(timestep):
    global coupling_strength_sum
    coupling_vect = np.empty(len(particle_list))
    # Can parallelize here
    stopnum = len(particle_list)
    for i in range(stopnum):
        # print(HH_advection_velocity(particle_list[i].mu))

        coupling_vect[i] = upai(timestep, timestep, particle_list[i])
    coupling_strength_sum = np.sum(coupling_vect)


def split_particle_in_direction(p, direction):
    omega = 0.21921 # Found by optimization in eqn. 14
    a = 1.03332
    p_center = Particle(p.weight * (1 - 2 * omega))
    p_left = Particle(p.weight * omega)
    p_right = Particle(p.weight * omega)
    p_center.mu = p.mu
    p_left.mu = p.mu - a * direction
    p_right.mu = p.mu + a * direction

    N = np.zeros((DIM, DIM))
    normalizer = np.dot(direction, direction)
    for i in range(DIM):
        # Eqn. 15 in paper
        N[i, :] = p.M[i, :] - (1.0 - 1.0 / math.sqrt(2)) * np.dot(direction, p.M[i, :]) / normalizer * direction
    
    p_center.M = N
    p_left.M = copy.deepcopy(N)
    p_right.M = copy.deepcopy(N)

    global KD
    distance_A, index_A = KD.query(p_left.mu)
    phase_A = 2.0 * np.pi * index_A / len(limit_cycle)
    distance_B, index_B = KD.query(p_center.mu)
    phase_B = 2.0 * np.pi * index_B / len(limit_cycle)
    distance_C, index_C = KD.query(p_right.mu)
    phase_C = 2.0 * np.pi * index_C / len(limit_cycle)

    # print("\nWeight before split:", p.weight)
    # print(f"Phases of center, left, right: {phase_A:.3f} {phase_B:.3f} {phase_C:.3f}")
    # print(f"Space distances: {np.linalg.norm(p_left.mu-p_center.mu):.3f}, {np.linalg.norm(p_left.mu-p_right.mu):.3f}")
    # print(f"Phase differences: {phase_A-phase_B:.3f}, {phase_A-phase_C:.3f}")
    # print(f"Distances from limit cycle: {distance_A:.3f} {distance_B:.3f} {distance_C:.3f}")

    return p_center, p_left, p_right

    # direction_outer_prod = np.matmul(direction, direction.T)
    # p_center.M = np.matmul(p.M, p.M.T) - direction_outer_prod


def combine_particles():
    global particle_list
    # tau = 0.0001
    combine_radius = 0.02
    new_particle_list = []
    blocksize = 0.04 # Edge length of blocks in the grid
    blockdict = {} # Maps a block ID to a vector of particles inside the block

    # Find min and max in each dimension. We then use these as bounds for the grid.
    min_coords, max_coords = np.copy(particle_list[0].mu), np.copy(particle_list[0].mu)
    block_counts = np.empty(DIM)
    for i in range(DIM):
        for p in particle_list:
            min_coords[i] = min(min_coords[i], p.mu[i])
            max_coords[i] = max(max_coords[i], p.mu[i])
        block_counts[i] = 2 + int((max_coords[i] - min_coords[i]) / blocksize)
    
    # Function which maps a particle location to the enumeration of its corresponding block.
    def location_to_id(mu):
        id = 0
        for i in range(DIM):
            id *= block_counts[i]
            ith_index = (int) ((mu[i] - min_coords[i]) / blocksize)
            id += ith_index
        return id

    for p in particle_list:
        id = location_to_id(p.mu)
        if id in blockdict:
            blockdict[id].append(p)
        else:
            blockdict[id] = [p]
    
    # print("Blockdict len:", len(blockdict))
    # print("Num particles:", len(particle_list))
    # print("The weights:", *[x.weight for x in particle_list])

    # Try doing combines in each grid block which actually contains particles.
    for _, block_particles in blockdict.items():
        local_count = len(block_particles)
        combined_yet = [False for i in range(local_count)]
        # O(n^2) search loop, but within each grid.
        oldfat = 0.0
        for i in range(local_count):
            if combined_yet[i]:
                continue
            combined_yet[i] = True
            combine_list = [block_particles[i]]
            # oldfat += block_particles[i].weight
            for j in range(local_count):
                if combined_yet[j]:
                    continue
                dist_ij = np.linalg.norm(block_particles[i].mu - block_particles[j].mu)
                if dist_ij <= combine_radius:
                    combined_yet[j] = True
                    combine_list.append(block_particles[j])
                    # oldfat += block_particles[j].weight
            combined_particle = combine_specific_particles(combine_list)
            # print("Extra fat:", combined_particle.weight, oldfat)
            new_particle_list.append(combined_particle)

    # print("Old particle list length:", len(particle_list))
    # print("New particle list length:", len(new_particle_list))

    particle_list = new_particle_list


def combine_specific_particles(combine_list):
    if len(combine_list) == 1:
        return combine_list[0]
    #print("Num to be combined:", len(combine_list))
    big_p = Particle(0.0)
    sum_weight = 0.0
    sum_mu = np.zeros(DIM)
    sum_M = np.zeros((DIM, DIM))
    for p in combine_list:
        sum_weight += p.weight
        sum_mu += p.weight * p.mu
    big_p.weight = sum_weight
    big_p.mu = sum_mu / sum_weight
    for p in combine_list:
        center_diff = p.mu - big_p.mu
        sum_M += p.weight / sum_weight * (p.M + np.matmul(center_diff, center_diff.T))
    big_p.M = sum_M
    return big_p


####### Plotting functions #######

# Initializing projection plane for density.
np.random.seed(seed=int(time.time()))
v1, v2 = np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])
# v1, v2 = np.random.rand(DIM), np.random.rand(DIM)
v2 = v2 - v1 * (
    v1.dot(v2) / np.linalg.norm(v1) ** 2
)  # Component of v2 orthogonal to v1.
v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)  # Normalize v1 and v2.
v_proj = np.stack([v1, v2], axis=-1)

# General plotting parameters.
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.dpi"] = 300
axis_names = [
    "Membrane potential / 100 (mV)",
    "Na$\mathregular{^+}$ subunit activation",
    "K$\mathregular{^+}$ subunit activation",
    "Na$\mathregular{^+}$ subunit inactivation",
    # "Non-inactivating K$\mathregular{^+}$",
]


# Density plot parameters.
projdims = [0, 1]

inc = 0.005
rangeX = [-0.25, 1.25]
rangeY = [-0.05, 1.05]
grid_stepsX = round((rangeX[1] - rangeX[0]) / inc)
grid_stepsY = round((rangeY[1] - rangeY[0]) / inc)

x0 = np.arange(rangeX[0], rangeX[1], inc) # Negative range for voltage hyperpolarization.
x1 = np.arange(rangeY[1], rangeY[0], -inc) # Flipped for right-side-up plotting.
xx0, xx1 = np.meshgrid(x0, x1)
xx0, xx1 = xx0.ravel(), xx1.ravel()
mesh = np.stack([xx0, xx1]).T

# Scatter plot parameters.
projdims2 = [0, 1, 2]


def calculate_mesh_density(weights, centers, cov_matrices):
    # weights = 0.20 * np.ones(len(centers))
    gaussians = []
    for X, M in zip(centers, cov_matrices):
        # Projection onto the 2D plane defined by 2 vectors in R^DIM?
        M = v_proj.T.dot(M).dot(v_proj)
        # For 2D projection, we extract the 2*2 relevant entries of the covariance matrix.
        # gaussian = multivariate_normal(mean=X[projdims], cov=M[projdims].T[projdims])
        gaussian = multivariate_normal(mean=X[projdims], cov=M)
        gaussians.append(gaussian)

    result = sum(
        weight * gaussian.pdf(mesh) for weight, gaussian in zip(weights, gaussians)
    )
    result = np.power(result, 0.25)
    return result.reshape(grid_stepsY, grid_stepsX)

def single_update_plot(p_list, t, KD):
    global T
    global limit_cycle
    # Modified to take input from a list of centers
    # Arrays for the center location, weight, and covariance matrix of every particle.
    weights = np.array([p.weight for p in p_list])
    centers = np.array([p.mu for p in p_list])
    cov_matrices = np.array([np.matmul(p.M, p.M.T) for p in p_list])
    density_data = calculate_mesh_density(weights, centers, cov_matrices)

    fig = plt.figure()

    plt.suptitle(f"Step {t}")

    # Density plot.
    p1 = fig.add_subplot(1, 2, 1)
    # p1.set_xlabel("Projection onto " + np.array2string(v1, precision=2))
    # p1.set_ylabel("Projection onto " + np.array2string(v2, precision=2))
    p1.set_xlabel(axis_names[projdims[0]])
    p1.set_ylabel(axis_names[projdims[1]])
    p1.set_xticks(
        [0, grid_stepsX - 1], rangeX
    )
    p1.set_yticks([0, grid_stepsY - 1], rangeY[::-1])
    avg_V = np.sum(np.multiply(centers[:, 0], weights))
    p1.set_title(
        f"Average voltage * 100 (mV): {int(avg_V * 100)}"
    )
    p1.imshow(density_data, cmap=plt.cm.get_cmap("magma"))

    # Scatter plot.
    p2 = fig.add_subplot(1, 2, 2, projection="3d")
    p2.set_title(f"Particle count: {len(p_list)}")
    sc = p2.scatter(
        centers.T[0],
        centers.T[1],
        centers.T[2],
        s=300 * weights,
        facecolors="none",
        edgecolors="blue",
    )
    # fig.colorbar(sc, ax=p2, fraction=0.03, pad=0.15)
    p2.set_xlim3d(x0[0], x0[-1] + inc)
    p2.set_ylim3d(0, 1)
    p2.set_zlim3d(0, 1)
    p2.set_xlabel(axis_names[projdims2[0]])
    p2.set_ylabel(axis_names[projdims2[1]])
    p2.set_zlabel(axis_names[projdims2[2]])


def find_limit_cycle():
    def integrate(h, y0, f):
        try:
            y = np.array(y0, dtype=float).copy()
            t = 0
            while True:
                # Runge-Kutta 4th order
                k1 = f(y)
                k2 = f(y + 0.5 * h * k1)
                k3 = f(y + 0.5 * h * k2)
                k4 = f(y + h * k3)
                y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                t += h
                yield t, y.copy()
        except ValueError:
            print(y0)

    def shot(z1):
        h = 1e-3
        t = 0
        y0 = z1
        last_points = [(t, y0)]
        for t, y in integrate(h, y0, HH_advection_velocity):
            last_points.append((t, y))
            if len(last_points) > 4:
                last_points = last_points[-4:]
            
            if len(last_points) < 4:
                continue
                
            (t1, y1), (t2, y2), (t3, y3), (t4, y4) = last_points
            if y2[0] < 0 and y3[0] >= 0:
                zfA = lagrange([y1[0], y2[0], y3[0], y4[0]], [y1[1], y2[1], y3[1], y4[1]])
                zfB = lagrange([y1[0], y2[0], y3[0], y4[0]], [y1[2], y2[2], y3[2], y4[2]])
                zfC = lagrange([y1[0], y2[0], y3[0], y4[0]], [y1[3], y2[3], y3[3], y4[3]])
                tf = lagrange([y1[0], y2[0], y3[0], y4[0]], [t1, t2, t3, t4])
                return tf(0), np.array([0, zfA(0), zfB(0), zfC(0)])

    z = np.array([0,0.0501172,0.40763994,0.44179683])
    for i in range(20):
        TT, zn = shot(z)
        dz = zn - z
        z = zn    
        print(f'Iteration {i}, TT = {TT:10.6f}, z = {z}, delta z = {dz}')
        if abs(np.linalg.norm(dz)) < 1e-15:
            break

    ts = []
    ys = []
    for t, y in integrate(1e-3, z, HH_advection_velocity):
        if t > TT:
            break
        ts.append(t)
        ys.append(y)
        
    ys = np.array(ys)
    return ys


if __name__ == "__main__":

    # Update 2024/09/04: manually set these.
    if len(sys.argv) < 4:
        print("Error: need to give max_steps, diff_coeff, coupling_coeff")
        exit(1)
    max_steps = int(sys.argv[1])
    diff_coeff = float(sys.argv[2])
    coupling_coeff = float(sys.argv[3])

    if len(sys.argv) > 4: # If we have save file:
        savefile = np.load(sys.argv[4])
        T = int(sys.argv[3].split('.')[0][-6:])
        voltage_record = savefile["voltage_record"]
        num_particles = len(savefile["mu_array"])
        for i in range(num_particles):
            particle_list.append(Particle(savefile["w_array"][i]))
            particle_list[-1].mu = savefile["mu_array"][i]
            particle_list[-1].M = savefile["M_array"][i]
        timestep = 0.05

    else:
        T = 0
        voltage_record = np.array([])
        num_particles = 1
        timestep = 0.05
        init_weights = np.random.rand(num_particles)
        init_weights /= np.sum(init_weights)

        # Initialize some particles
        for i in range(num_particles):
            particle_list.append(Particle(init_weights[i]))

        particle_list[0].mu = np.array([0.5, 0.5, 0.5, 0.5])
        # particle_list[0].mu = np.array([0.03691094, 0.07652289, 0.39008829, 0.47730723])
        particle_list[0].M *= 4

    # 2024/09/09: Track the number of particles over time.
    particle_history = []

    limit_cycle = np.copy(find_limit_cycle())
    KD = KDTree(limit_cycle)

    remaining_time = max_steps - T
    start_time = time.time()
    for _ in range(remaining_time):

        T += 1
        # print("Step", T)

        update_ODE_adaptive_split(timestep)
        combine_particles()

        avg_V = np.sum(np.multiply(np.array([p.mu for p in particle_list])[:, 0], 
                                   np.array([p.weight for p in particle_list])))
        voltage_record = np.append(voltage_record, avg_V)

        particle_history.append(len(particle_list))

        # if T % 5 == 0:
        #     single_update_plot(particle_list, T)
        #     plt.savefig(str(T).zfill(6) + ".jpeg")
        #     plt.close()

        if T % 5 == 0:
            print(f"diff_coeff={diff_coeff:.2e}, coupling_coeff={coupling_coeff:.2e}, got to step {T} of {max_steps}")
        #     save_name = f"backup_{str(T).zfill(6)}.npz"
        #     mu_array = np.array([p.mu for p in particle_list])
        #     w_array = np.array([p.weight for p in particle_list])
        #     M_array = np.array([p.M for p in particle_list])
        #     np.savez(save_name, mu_array=mu_array, w_array=w_array, M_array=M_array, voltage_record=voltage_record)

        #     fig = plt.figure()
        #     fig.suptitle("Average voltage over time")
        #     plot_V = fig.add_subplot(1, 1, 1)
        #     plot_V.plot(range(T), voltage_record)
        #     plt.savefig(f"voltage-record_{T}.png")
        #     plt.close()

    total_time = time.time() - start_time
    output_name = f"phasereduction-output_{diff_coeff:.2e}_{coupling_coeff:.2e}_{max_steps}.npz"
    time_array = np.array([total_time]) # just one number (total time of the simulation)
    pcount_array = np.array(particle_history)
    diff_array = np.array([diff_coeff])
    coup_array = np.array([coupling_coeff])
    np.savez(output_name, time_array=time_array, pcount_array=pcount_array,
            diff_array=diff_array, coup_array=coup_array, voltage_record=voltage_record)

    summary = f"diff_coeff={diff_coeff:.2e}, coupling_coeff={coupling_coeff:.2e}"
    print(summary)

    fig = plt.figure()
    fig.suptitle(summary)
    plot_V = fig.add_subplot(1, 1, 1)
    plot_V.plot(range(T), voltage_record)
    plot_V.set_xlabel("Timestep")
    plot_V.set_ylabel("Average normalized voltage (mV/100)")
    plot_V.grid()
    plt.savefig(f"phasereduction-voltage_{appcur:.2e}_{diff_coeff:.2e}_coupling_{coupling_coeff:.2e}_{max_steps}.png")
    plt.close()

    # video_name = "hh4d_" + str(date.today()) + "_" + str(time.time()) + ".mp4"
    # subprocess.call(
    #     [
    #         "ffmpeg",
    #         "-framerate",
    #         "15",
    #         "-pattern_type",
    #         "glob",
    #         "-i",
    #         "*.jpeg",
    #         "-r",
    #         "30",
    #         "-pix_fmt",
    #         "yuv420p",
    #         video_name,
    #     ]
    # )