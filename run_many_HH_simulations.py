import sys
import os
import subprocess
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime

# Set our parameter lists here:
diff_coeffs = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4])
coupling_coeffs = np.array([0.01, 0.05, 0.10, 0.25, 0.50])
# diff_coeffs = np.array([1e-6, 1e-5])
# coupling_coeffs = np.array([0.25, 0.50])
N, M = len(diff_coeffs), len(coupling_coeffs)
max_steps = 3000


# Run many simulations with varied noise and coupling coefficients.
# Each simulation gets its own thread (can run them in parallel, but choose not to here).
# We move on after all of the simulations/threads are finished.

# Here is a helper function for running one simulation:
def run_subprocess(D, C):
    cmd = ['python3', 'HH_manual_noise_coupling.py',
           str(max_steps), str(D), str(C)]
    print(f"Simulating with D={D:.2e}, C={C:.2e}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Read stdout in real-time
    for stdout_line in process.stdout:
        print(f"Subprocess (D={D}, C={C}) Output: {stdout_line}", end='')

    # Wait for the process to finish and check for errors
    process.wait()
    stderr = process.stderr.read()
    if stderr:
        print(f"Subprocess (D={D}, C={C}) Error: {stderr}", end='')


# Here is the double for loop which runs all simulations with different noise/coupling.
# Better to just run the simulations sequentially, instead of in parallel...
for D in diff_coeffs:
    for C in coupling_coeffs:
        # continue
        run_subprocess(D, C)

print("Finished all simulations")


# Output file format should have diff_coeff, coupling_coeff, and max_steps:
#     particlecounts_1.00e-5_1.00e-01_5000.npz


# General plotting parameters.
plt.rcParams["figure.figsize"] = [20, 15]
plt.rcParams["figure.dpi"] = 300
fig, axes = plt.subplots(N, M, constrained_layout=True)


# Now all simulations should be done.
# Make a grid of N * M plots of particle counts, for all simulations.
for filename in os.listdir("."):
    if not filename.endswith(f".npz"):
        continue
    file = np.load(filename)
    D = file["diff_array"][0]
    C = file["coup_array"][0]

    print(D, C)

    # Find the proper indices of the file we loaded (may be out of order)
    i = np.where(np.isclose(diff_coeffs, D, 1e-10))[0][0]
    j = np.where(np.isclose(coupling_coeffs, C, 1e-10))[0][0]

    print(i, j)

    # Now plot in the proper subplot.
    pcounts = file["pcount_array"]
    p1 = axes[i, j]
    # p1.set_xlabel("Timesteps")
    p1.set_ylabel("# of particles")

    # Get average particle count, but exclude steps up to 10.
    avg_pcount = np.mean(pcounts[0:])
    p1.set_title(f"D={D:.1e}, C={C:.1e}, p_avg={avg_pcount:.1f}")
    print(avg_pcount, pcounts)
    p1.plot(pcounts)
    # p1.set_xticks(range(len(pcounts)))
    p1.grid()


fig.tight_layout(pad=0.4) # Stops the subplots from being too close together
fig.savefig("all_particle_counts.jpeg")

fig = plt.figure(num=1, clear=True)
fig, axes = plt.subplots(N, M, constrained_layout=True)

# We can do the exact same thing for voltages...
for filename in os.listdir("."):
    if not filename.endswith(f".npz"):
        continue
    file = np.load(filename)
    D = file["diff_array"][0]
    C = file["coup_array"][0]

    print(D, C)

    # Find the proper indices of the file we loaded (may be out of order)
    i = np.where(np.isclose(diff_coeffs, D, 1e-10))[0][0]
    j = np.where(np.isclose(coupling_coeffs, C, 1e-10))[0][0]

    print(i, j)

    # Now plot in the proper subplot.
    pcounts = file["pcount_array"]
    voltages = file["voltage_record"]
    p1 = axes[i, j]
    p1.set_xlabel("Timesteps")
    p1.set_ylabel("Average normalized voltage (mV/100)")

    # Get average particle count, but exclude steps up to 10.
    avg_pcount = np.mean(pcounts[0:])
    p1.set_title(f"diffusion coeff={D:.1e}, coupling coeff={C:.1e}")
    print(avg_pcount, pcounts)
    p1.plot(voltages)
    # p1.set_xticks(range(len(pcounts)))
    p1.grid()


fig.tight_layout(pad=0.4) # Stops the subplots from being too close together
fig.savefig("all_voltage_records.jpeg")
plt.close()