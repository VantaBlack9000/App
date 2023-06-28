#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sko.GA import GA_TSP
from scipy import spatial
from scipy.spatial import distance
from tqdm import tqdm

def generate_random_coordinates(num_points):
    points_coordinate = np.random.rand(num_points, 2)
    return pd.DataFrame(points_coordinate, columns=["lat", "lon"])

def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

problem_sizes = range(3, 101)
max_iters = [100, 200, 300]
size_pops = [50, 100, 150]
prob_muts = [0.1, 0.5, 1.0]

results = []

total_iterations = len(problem_sizes) * len(max_iters) * len(size_pops) * len(prob_muts)
pbar = tqdm(total=total_iterations)  # Initialize progress bar

for problem_size in problem_sizes:
    coords = generate_random_coordinates(problem_size)
    points_coordinate = coords[["lat", "lon"]].to_numpy()
    distance_matrix = distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

    best_result = None
    best_distance = float('inf')

    for max_iter in max_iters:
        for size_pop in size_pops:
            for prob_mut in prob_muts:
                ga_tsp = GA_TSP(
                    func=cal_total_distance,
                    n_dim=problem_size,
                    size_pop=size_pop,
                    max_iter=max_iter,
                    prob_mut=prob_mut
                )
                best_points, best_distance_ga = ga_tsp.run()
                if best_distance_ga < best_distance:
                    best_result = {
                        "problem_size": problem_size,
                        "max_iter": max_iter,
                        "size_pop": size_pop,
                        "prob_mut": prob_mut,
                        "best_distance_ga": best_distance_ga,
                    }
                    best_distance = best_distance_ga
                pbar.update(1)  # Update progress bar

    results.append(best_result)

pbar.close()  # Close the progress bar

# Generate performance plots
x_ticks = [result["problem_size"] for result in results]
y_time = [result["total_time_ga"] for result in results]
y_distance = [result["best_distance_ga"] for result in results]

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(x_ticks, y_time, marker='o')
ax[0].set_xlabel("Problem Size")
ax[0].set_ylabel("Execution Time (s)")
ax[0].set_title("Execution Time vs. Problem Size")

ax[1].plot(x_ticks, y_distance, marker='o')
ax[1].set_xlabel("Problem Size")
ax[1].set_ylabel("Best Distance")
ax[1].set_title("Best Distance vs. Problem Size")

plt.tight_layout()
plt.show()

for result in results:
    print(f"Problem Size: {result['problem_size']}")
    print(f"Best Parameters: Max Iter: {result['max_iter']}, Size Pop: {result['size_pop']}, Prob Mut: {result['prob_mut']}")
    print(f"Best Distance: {result['best_distance_ga']}")
    print()